"""Synchronized Tsit5 schedule and fixed-step buffer/stage experiments."""
import numpy as np
import cupy as cp
import math
from rc_gpu_integrators import IntegratorGPU, TS_A, TS_C

class FixedGPU(IntegratorGPU):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.swap='swap' in self.variant
        self.const='const' in self.variant
        self.scheduled='schedule' in self.variant
        source=self.source
        if self.scheduled:
            p=self.p.get()
            multiplier=1.2 if 'gentle' in self.variant else 1.5
            self.schedule_center,self.schedule_sigma,self.schedule_multiplier=p[0],p[1],multiplier
            old='double t=clock[0], nominal=clock[3];'
            new=(f'double t=clock[0], nominal=clock[3]; '
                 f'if (fabs((t-({p[0]:.17g}))/({p[1]:.17g}))>0.8) nominal*={multiplier:.17g};')
            assert old in source
            source=source.replace(old,new)
        params=','.join(f'const STATE* k{i}' for i in range(7))
        source+='\nextern "C" {\n'
        source+=f'__global__ void stage_ptr(STATE* dest,const STATE* base,{params},const double* w,const double* clock,long D,int row,int terms) {{'
        source+='long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;'
        source+='const STATE* ks[7]={k0,k1,k2,k3,k4,k5,k6};double s=0.;'
        source+='for(int k=0;k<terms;k++)if(w[row*7+k]!=0.)s+=w[row*7+k]*(double)ks[k][x];'
        source+='dest[x]=(double)base[x]+clock[1]*s;}\n'
        for row in range(1,7):
            source+=f'__global__ void stage_c{row}(STATE* dest,const STATE* base,{params},const double* clock,long D) {{'
            source+='long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;double s=0.;'
            for k,a in enumerate(TS_A[row]):
                if a:source+=f's+=({a:.17g})*(double)k{k}[x];'
            source+='dest[x]=(double)base[x]+clock[1]*s;}\n'
        source+='}\n'
        self.source=source
        module=cp.RawModule(code=source,options=('--std=c++14','-DSTATE=double'))
        names=list(self.kernels)+['stage_ptr']+[f'stage_c{i}' for i in range(1,7)]
        self.kernels={name:module.get_function(name) for name in names}

    def fixed_launches(self,start,target):
        if not self.scheduled:return super().fixed_launches(start,target)
        # Plan the synchronized schedule, including switch and output clipping.
        # This is clock arithmetic only; no trajectory/error information is read.
        t=start;steps=0
        while t<target-1e-15:
            dt=self.dt*(self.schedule_multiplier if abs((t-self.schedule_center)/self.schedule_sigma)>.8 else 1.)
            edge=(math.floor((t+1e-14)/2.5e-7)+1)*2.5e-7
            h=max(0.,min(dt,edge-t,target-t))
            if h<=0.:raise RuntimeError('Scheduled clock stalled')
            t+=h;steps+=1
        return math.ceil(steps/self.chunk)+2

    def block(self,steps=16):
        if not (self.swap or self.const):return super().block(steps)
        assert self.tsit and not self.adaptive and not self.fused
        y,yn=self.y,self.yn
        ks=list(self.ks)
        for _ in range(steps):
            self.kernels['begin_step']((1,),(1,),(self.clock,))
            if self.swap:
                # Read the previous final derivative directly; no FSAL copy.
                ks[0],ks[6]=ks[6],ks[0]
            else:self.launch('start_fsal',(self.stages,self.clock,np.int64(2*self.size)),2*self.size)
            stage_y=y
            for k in range(7):
                self.rhs_new(stage_y,ks[k],TS_C[k],skip=k==0)
                if k<6:
                    dest=yn if k==5 else self.temps[k%2]
                    if self.const:
                        self.launch(f'stage_c{k+1}',(dest,y,*ks,self.clock,np.int64(2*self.size)),2*self.size)
                    else:
                        self.launch('stage_ptr',(dest,y,*ks,self.weights,self.clock,np.int64(2*self.size),np.int32(k+1),np.int32(k+1)),2*self.size)
                    stage_y=dest
            self.kernels['control']((1,),(1,),(self.clock,self.errors,np.int32(self.batch),np.int32(0),np.int32(1)))
            if self.swap:y,yn=yn,y
            else:self.launch('accept_step',(y,yn,self.clock,np.int64(2*self.size)),2*self.size)
        # Captured graphs have an even number of steps, so both buffer and
        # derivative permutations return to their original addresses. Warmup
        # uses one step but discards its result before capture.
        if steps>1:assert steps%2==0
