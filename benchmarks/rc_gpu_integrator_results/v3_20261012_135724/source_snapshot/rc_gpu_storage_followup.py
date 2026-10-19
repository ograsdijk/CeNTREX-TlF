"""Graph tails, GPU population history, packed Hermitian and IF experiments."""
import math
import time
from pathlib import Path
import numpy as np
import cupy as cp
from rc_gpu_fixed_followup import FixedGPU
from rc_gpu_integrators import TS_C

class OutputGPU(FixedGPU):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.exact='exact' in self.variant
        self.devicepop='devicepop' in self.variant
        self.packed=False
        self.source+='''
extern "C" {
__global__ void set_end(double* clock,double target){clock[2]=target;}
__global__ void sample_pop(const double* y,double* pops,double* times,const double* clock,int B,int n,int sample,int packed){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(x<B*n){int b=x/n,i=x%n;long ij=packed?i*n-i*(i-1)/2:i*n+i;
 pops[(long)sample*B*n+x]=y[ij*B+b];}
 if(x==0)times[sample]=clock[0];
}
}
'''
        self._compile()

    def _compile(self):
        module=cp.RawModule(code=self.source,options=('--std=c++14','-DSTATE=double'))
        names=list(self.kernels)+['set_end','sample_pop']
        self.kernels={name:module.get_function(name) for name in dict.fromkeys(names)}

    def planned_steps(self,start,target):
        t=start;steps=0
        while t<target-1e-15:
            factor=(self.schedule_multiplier if self.scheduled and
                    abs((t-self.schedule_center)/self.schedule_sigma)>.8 else 1.)
            edge=(math.floor((t+1e-14)/2.5e-7)+1)*2.5e-7
            h=max(0.,min(self.dt*factor,edge-t,target-t))
            if h<=0.:raise RuntimeError('Clock planner stalled')
            t+=h;steps+=1
        return steps

    def capture(self,steps=None,rhs_only=False):
        super().capture(steps,rhs_only)
        self.tail_graphs={self.chunk:self.graph}
        if self.exact:
            for count in range(2,self.chunk,2):
                stream=cp.cuda.Stream(non_blocking=True)
                with stream:
                    stream.begin_capture();self.block(count);graph=stream.end_capture()
                self.tail_graphs[count]=graph
            self.reset();cp.cuda.get_current_stream().synchronize()

    def launch_to(self,start,target):
        self.kernels['set_end']((1,),(1,),(self.clock,np.float64(target)))
        if self.exact:
            count=self.planned_steps(start,target)
            full,tail=divmod(count,self.chunk)
            for _ in range(full):self.graph.launch()
            if tail:self.tail_graphs[2*math.ceil(tail/2)].launch()
        else:
            for _ in range(self.fixed_launches(start,target)):self.graph.launch()

    def evolve(self,repeats=3):
        if not self.exact:return super().evolve(repeats)
        timings=[];stats=[]
        for _ in range(repeats):
            self.reset();cp.cuda.get_current_stream().synchronize();tick=time.perf_counter()
            self.launch_to(self.start,self.end)
            clock=self.clock.get()
            if abs(clock[0]-self.end)>1e-13:raise RuntimeError('Exact tail missed endpoint')
            timings.append(time.perf_counter()-tick)
            stats.append(dict(accepted=int(clock[6]),rejected=int(clock[7]),final_time=float(clock[0])))
        self.stats=stats
        return self.download().astype(np.complex128),timings,float(clock[0])

    def full_transit(self):
        self.reset();cp.cuda.get_current_stream().synchronize();tick=time.perf_counter()
        targets=np.linspace(0.,self.end,201)
        if self.devicepop:
            pops=cp.empty((201,self.batch,self.n),cp.float64);times=cp.empty(201,cp.float64)
            def sample(index):
                self.launch('sample_pop',(self.y,pops,times,self.clock,np.int32(self.batch),np.int32(self.n),
                                         np.int32(index),np.int32(self.packed)),self.batch*self.n)
        else:
            hostpops=[]
            def sample(index):
                hostpops.append(np.diagonal(self.download(),axis1=-2,axis2=-1).real.copy())
        sample(0)
        for index,target in enumerate(targets[1:],1):
            self.launch_to(float(targets[index-1]),float(target))
            if not self.devicepop:
                if abs(self.clock.get()[0]-target)>1e-13:raise RuntimeError('Population endpoint missed')
            sample(index)
        exit_rho=self.download();exit_wall=time.perf_counter()-tick
        if self.devicepop:
            actual_times=times.get()
            if np.max(np.abs(actual_times-targets))>1e-13:raise RuntimeError('GPU population sample time mismatch')
            populations=pops.get().transpose(1,0,2).copy()
        else:populations=np.stack(hostpops,axis=1)
        p=self.p.get();p[2:5]=0.;self.p.set(p)
        clock=self.clock.get();clock[4]=0.;self.clock.set(clock)
        from centrex_tlf import hamiltonian
        end=self.end+self.context['physics']['post_decay_lifetimes']/float(hamiltonian.Γ)
        self.launch_to(self.end,end)
        final=self.download()
        if abs(self.clock.get()[0]-end)>1e-13:raise RuntimeError('Dark endpoint missed')
        return np.stack((exit_rho,final),axis=1),populations,dict(
            field_exit_wall_s_including_sampling=exit_wall,
            full_wall_s_including_sampling_and_post_decay=time.perf_counter()-tick)

class PackedGPU(OutputGPU):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.packed=True;self.direct='direct' in self.variant
        self.upper=np.triu_indices(self.n);self.size=self.batch*len(self.upper[0])
        self.y=self.encode(self.initial_host);self.initial=self.y.copy()
        self.stages=cp.zeros((7,2,self.size),cp.float64);self.ks=[self.stages[k] for k in range(7)]
        self.temps=[cp.empty_like(self.y),cp.empty_like(self.y)];self.yn=cp.empty_like(self.y)
        if hasattr(self,'temp'):del self.temp
        self.prows=cp.asarray(self.upper[0].astype(np.int32));self.pcols=cp.asarray(self.upper[1].astype(np.int32))
        self.source+=(Path(__file__).parent/'rc_gpu_packed.cu').read_text()
        self._compile()
        module=cp.RawModule(code=self.source,options=('--std=c++14','-DSTATE=double'))
        for name in ['packed_product','packed_finish','packed_direct']:self.kernels[name]=module.get_function(name)
        self.kernels['product']=self.kernels['packed_product']
        if self.direct:self.ar=cp.empty(1,cp.float64);self.ai=cp.empty(1,cp.float64)

    def encode(self,host):
        if np.max(np.abs(host-host.conj().swapaxes(-1,-2)))>1e-12:raise ValueError('Non-Hermitian initial density')
        a=host[:,self.upper[0],self.upper[1]].T
        return cp.asarray(np.ascontiguousarray(np.stack((a.real,a.imag)).reshape(2,-1)))

    def download(self,y=None):
        a=cp.asnumpy(self.y if y is None else y)
        z=(a[0]+1j*a[1]).reshape(-1,self.batch).T
        out=np.zeros((self.batch,self.n,self.n),complex)
        out[:,self.upper[0],self.upper[1]]=z
        out[:,self.upper[1],self.upper[0]]=z.conj()
        return out

    def rhs_new(self,y,out,fraction,skip=False,next_stage=None,row=0,current=0,build=True):
        assert next_stage is None
        if build:
            self.kernels['coefficients']((1,),(1,),(self.clock,self.p,np.float64(fraction),self.cr,self.ci))
            self.launch('build_h',(self.positions,self.br,self.bi,self.diag,self.cr,self.ci,self.hz,self.hr,self.hi,
                                  np.int32(self.positions.size),np.int32(self.n),np.int32(0)),self.positions.size)
        tail=(self.diag,self.mask,self.det,self.rates,self.tp,self.tc,self.tr,self.prows,self.pcols,
              np.int32(self.batch),np.int32(self.n),self.clock,np.int32(skip))
        if self.direct:self.launch('packed_direct',(y,out,self.ptr,self.col,self.hr,self.hi,*tail))
        else:
            self.launch('product',(y,self.ar,self.ai,self.ptr,self.col,self.hr,self.hi,np.int32(self.batch),np.int32(self.n),self.clock,np.int32(skip)),self.batch*self.n*self.n)
            self.launch('packed_finish',(y,out,self.ar,self.ai,*tail))

class IntegratingFactorGPU(OutputGPU):
    """Local-step integrating factor for Hamiltonian diagonal and decay loss.

    rho(s)=exp(L0*s) w(s), where L0_ij=-i*(Hii-Hjj)-(rate_i+rate_j)/2.
    Sparse commutator and population transfers are evolved with fixed Tsit5.
    The coordinate origin resets each step, avoiding long-time exponentials.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        a=self.source.index('__global__ void finish(')
        b=self.source.index('__global__ void errors(',a)
        nonlinear=self.source[a:b].replace('void finish(','void finish_n(')
        old='double d=diag[i]-diag[j]+det[b]*(mask[i]-mask[j]),loss=-.5*(rates[i]+rates[j]);'
        assert old in nonlinear
        nonlinear=nonlinear.replace(old,'double d=0.,loss=0.;')
        self.source+='\nextern "C" {\n'+nonlinear+'}\n'
        self.source+=(Path(__file__).parent/'rc_gpu_integrating_factor.cu').read_text()
        self._compile()
        module=cp.RawModule(code=self.source,options=('--std=c++14','-DSTATE=double'))
        for name in ['finish_n','if_factors','if_rotate']:self.kernels[name]=module.get_function(name)
        self.fqr=cp.empty(self.batch*self.n,cp.float64);self.fqi=cp.empty_like(self.fqr);self.finv=cp.empty_like(self.fqr)
        self.physical=cp.empty_like(self.y)

    def rotate(self,src,dest,inverse):
        self.launch('if_rotate',(src,dest,self.fqr,self.fqi,self.finv,np.int32(self.batch),np.int32(self.n),np.int32(inverse)))

    def block(self,steps=16):
        # No FSAL in the changing local coordinates. The final derivative is
        # unnecessary for fixed-step propagation, so six RHS evaluations remain.
        for _ in range(steps):
            self.kernels['begin_step']((1,),(1,),(self.clock,))
            state=self.y
            for k in range(6):
                if k:
                    self.launch('if_factors',(self.diag,self.mask,self.det,self.rates,self.clock,np.float64(TS_C[k]),
                                              self.fqr,self.fqi,self.finv,np.int32(self.batch),np.int32(self.n)),self.batch*self.n)
                    self.rotate(state,self.physical,False);physical=self.physical
                else:physical=state
                self.kernels['coefficients']((1,),(1,),(self.clock,self.p,np.float64(TS_C[k]),self.cr,self.ci))
                self.launch('build_h',(self.positions,self.br,self.bi,self.diag,self.cr,self.ci,self.hz,self.hr,self.hi,
                                      np.int32(self.positions.size),np.int32(self.n),np.int32(0)),self.positions.size)
                self.launch('product',(physical,self.ar,self.ai,self.ptr,self.col,self.hr,self.hi,
                                       np.int32(self.batch),np.int32(self.n),self.clock,np.int32(0)))
                self.launch('finish_n',(physical,self.ks[k],self.ar,self.ai,self.diag,self.mask,self.det,self.rates,self.tp,self.tc,self.tr,
                                       np.int32(self.batch),np.int32(self.n),self.clock,np.int32(0),self.yn,self.y,self.stages,self.weights,
                                       np.int32(0),np.int32(0),np.int32(0)))
                if k:self.rotate(self.ks[k],self.ks[k],True)
                dest=self.yn if k==5 else self.temps[k%2]
                self.launch(f'stage_c{k+1}',(dest,self.y,*self.ks,self.clock,np.int64(2*self.size)),2*self.size)
                state=dest
            # Stage 6 and the final solution share c=1, hence the factors above.
            self.rotate(self.yn,self.y,False)
            self.kernels['control']((1,),(1,),(self.clock,self.errors,np.int32(self.batch),np.int32(0),np.int32(0)))

def make_gpu(*args,**kwargs):
    variant=kwargs.get('variant',args[6] if len(args)>6 else 'base')
    integrator=kwargs.get('integrator',args[5] if len(args)>5 else 'tsit5')
    cls=PackedGPU if 'packed' in variant else (IntegratingFactorGPU if integrator=='if_tsit5' else OutputGPU)
    return cls(*args,**kwargs)
