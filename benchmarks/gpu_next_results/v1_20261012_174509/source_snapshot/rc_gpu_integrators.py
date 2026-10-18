"""GPU RK4/Tsit5 experiments and isolated optimization switches."""
from __future__ import annotations
import re
import time
from pathlib import Path
import cupy as cp
import numpy as np
from rc_gpu_sparse_methods import GPU

HERE = Path(__file__).resolve().parent
text = (HERE.parent/'rust/src/ode/tsit5.rs').read_text()
def constant(name):
    value = re.search(rf'const {name}: f64 = ([^;]+);', text)[1]
    return float(eval(value, {'__builtins__': {}}, {}))
TS_C = [0.] + [constant(f'C{i}') for i in range(2,8)]
TS_A = np.zeros((7,7))
for i in range(2,8):
    for j in range(1,i):TS_A[i-1,j-1] = constant(f'A{i}{j}')
TS_E = np.array([constant(f'E{i}') for i in range(1,8)])
RK_C = [0.,.5,.5,1.]
RK_A = np.zeros((5,7));RK_A[1,0]=.5;RK_A[2,1]=.5;RK_A[3,2]=1.;RK_A[4,:4]=[1/6,1/3,1/3,1/6]

class IntegratorGPU(GPU):
    def __init__(self, context,batch,start,end,dt_ns=1.,integrator='tsit5',
                 variant='base',chunk=16,atol=1e-10,rtol=1e-9,pattern='grid',velocity=150.):
        self.integrator=integrator;self.variant=variant;self.request_chunk=chunk
        self.adaptive=integrator=='adaptive_tsit5';self.tsit=integrator!='rk4'
        self.fused='fused' in variant or variant=='all'
        self.reuse='reuse' in variant or variant=='all'
        self.special='special' in variant or variant=='all'
        self.mixed='mixed' in variant
        self.atol,self.rtol=atol,rtol
        super().__init__(context,'staged_soa_minor',batch,start,end,250/dt_ns,velocity,pattern)
        self.dt=dt_ns*1e-9
        self.clock_initial=np.array([start,self.dt,end,self.dt,0.,0.,0.,0.])
        self.clock=cp.asarray(self.clock_initial)
        dtype=cp.float32 if self.mixed else cp.float64
        self.y=self.y.astype(dtype);self.initial=self.y.copy()
        self.stages=cp.zeros((7,2,self.size),dtype);self.ks=[self.stages[k] for k in range(7)]
        self.temps=[cp.empty_like(self.y),cp.empty_like(self.y)]
        self.yn=cp.empty_like(self.y);self.errors=cp.empty(batch,float)
        self.weights=cp.asarray(TS_A if self.tsit else RK_A);self.error_weights=cp.asarray(TS_E)
        source=(HERE/'rc_gpu_sparse_kernels.cu').read_text()
        # Only take generic clock, coefficient and Hamiltonian builders. The
        # new RHS preserves the existing staged sparse algebra exactly.
        source=source[:source.index('// Direct output ownership')]+'}\n'
        source+=(HERE/'rc_gpu_integrators.cu').read_text()
        if self.special:
            start_loop=source.index(' for(int p=ptr[i];p<ptr[i+1];p++)')
            end_loop=source.index('\n ar[x]=r;',start_loop)
            cases=[]
            for i in range(self.n):
                lines=[f'case {i}:']
                for p in range(self.f['ptr'][i],self.f['ptr'][i+1]):
                    c=self.f['columns'][p]
                    lines.append(f'{{long k=((long){c}*n+j)*B+b;double u=y[k],v=y[N+k];r+=hr[{p}]*u-hi[{p}]*v;w+=hr[{p}]*v+hi[{p}]*u;}}')
                cases.append(''.join(lines)+'break;')
            source=source[:start_loop]+'switch(i){'+''.join(cases)+'}'+source[end_loop:]
        options=('--std=c++14','-DSTATE=float' if self.mixed else '-DSTATE=double')
        module=cp.RawModule(code=source,options=options)
        self.kernels={k:module.get_function(k) for k in ['begin_step','coefficients','build_h','stage',
             'start_fsal','product','finish','errors','control','accept_step']}
        self.source=source
        self.eager=False

    def stage(self,dest,row,terms):
        self.launch('stage',(dest,self.y,self.stages,self.weights,self.clock,np.int64(2*self.size),np.int32(row),np.int32(terms)),2*self.size)

    def rhs_new(self,y,out,fraction,skip=False,next_stage=None,row=0,current=0,build=True):
        if build:
            self.kernels['coefficients']((1,),(1,),(self.clock,self.p,np.float64(fraction),self.cr,self.ci))
            self.launch('build_h',(self.positions,self.br,self.bi,self.diag,self.cr,self.ci,self.hz,self.hr,self.hi,
                          np.int32(self.positions.size),np.int32(self.n),np.int32(0)),self.positions.size)
        self.launch('product',(y,self.ar,self.ai,self.ptr,self.col,self.hr,self.hi,np.int32(self.batch),np.int32(self.n),self.clock,np.int32(skip)))
        self.launch('finish',(y,out,self.ar,self.ai,self.diag,self.mask,self.det,self.rates,self.tp,self.tc,self.tr,
             np.int32(self.batch),np.int32(self.n),self.clock,np.int32(skip),self.yn if next_stage is None else next_stage,
             self.y,self.stages,self.weights,np.int32(row),np.int32(current),np.int32(next_stage is not None)))

    def block(self,steps=16):
        count=7 if self.tsit else 4;fractions=TS_C if self.tsit else RK_C
        for _ in range(steps):
            self.kernels['begin_step']((1,),(1,),(self.clock,))
            if self.tsit:self.launch('start_fsal',(self.stages,self.clock,np.int64(2*self.size)),2*self.size)
            current_y=self.y
            for k in range(count):
                last=k==count-1
                next_y=None if last and self.tsit else (self.yn if k==count-2 and self.tsit or last else self.temps[k%2])
                row=k+1
                skip=self.tsit and k==0
                reuse=self.reuse and ((self.tsit and k==6) or (not self.tsit and k==2))
                self.rhs_new(current_y,self.ks[k],fractions[k],skip,next_y if self.fused else None,row,k,not reuse)
                if next_y is not None:
                    if not self.fused:self.stage(next_y,row,k+1)
                    current_y=next_y
            if self.adaptive:
                self.kernels['errors']((self.batch,),(256,),(self.y,self.yn,self.stages,self.error_weights,self.clock,
                                      self.errors,np.int32(self.batch),np.int32(self.n),np.float64(self.atol),np.float64(self.rtol)))
            self.kernels['control']((1,),(1,),(self.clock,self.errors,np.int32(self.batch),np.int32(self.adaptive),np.int32(self.tsit)))
            self.launch('accept_step',(self.y,self.yn,self.clock,np.int64(2*self.size)),2*self.size)

    def capture(self,steps=None,rhs_only=False):
        steps=self.request_chunk if steps is None else steps
        self.reset();self.block(1);cp.cuda.get_current_stream().synchronize();self.reset()
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture();self.block(steps);self.graph=stream.end_capture()
        self.chunk=steps;self.reset();cp.cuda.get_current_stream().synchronize()

    def evolve(self,repeats=3):
        times=[];stats=[]
        for _ in range(repeats):
            self.reset();cp.cuda.get_current_stream().synchronize();tick=time.perf_counter()
            if self.adaptive:
                attempts=0
                while True:
                    for j in range(4):self.graph.launch()
                    state=self.clock.get();attempts+=self.chunk*4
                    if state[0]>=self.end-1e-15:break
                    if attempts>3000000 or time.perf_counter()-tick>600:raise RuntimeError('Adaptive integration work limit')
            else:
                # Initial/final partial steps can add one attempt; all subsequent
                # switch intervals are an exact multiple only for some dt grids.
                launches=self.fixed_launches(self.start,self.end)
                for j in range(launches):self.graph.launch()
                state=self.clock.get()
                while state[0]<self.end-1e-15:
                    self.graph.launch();state=self.clock.get()
            cp.cuda.get_current_stream().synchronize();times.append(time.perf_counter()-tick)
            stats.append(dict(accepted=int(state[6]),rejected=int(state[7]),final_time=float(state[0])))
        self.stats=stats
        return self.download().astype(np.complex128),times,float(state[0])

    def rhs(self,y,out,fraction):
        self.rhs_new(y,out,fraction)

    def fixed_launches(self,start,target):
        return int(np.ceil((target-start)/self.dt/self.chunk))+2

    def full_transit(self):
        """Population history and both density endpoints with the new clock."""
        self.reset();started=time.perf_counter()
        populations=[np.diagonal(self.initial_host,axis1=-2,axis2=-1).real.copy()]
        def until(target):
            state=self.clock.get();state[2]=target;self.clock.set(state)
            if self.adaptive:
                attempts=0
                while state[0]<target-1e-15:
                    for _ in range(4):self.graph.launch()
                    state=self.clock.get();attempts+=self.chunk*4
                    if attempts>3000000:raise RuntimeError('Full-transit adaptive work limit')
            else:
                for _ in range(self.fixed_launches(float(state[0]),target)):self.graph.launch()
                state=self.clock.get()
                while state[0]<target-1e-15:self.graph.launch();state=self.clock.get()
            if abs(state[0]-target)>1e-13:raise RuntimeError('Incomplete full-transit output')
        for target in np.linspace(0.,self.end,201)[1:]:
            until(target);populations.append(np.diagonal(self.download(),axis1=-2,axis2=-1).real.copy())
        exit_rho=self.download();exit_wall=time.perf_counter()-started
        p=self.p.get();p[2:5]=0.;self.p.set(p)
        state=self.clock.get();state[4]=0.;self.clock.set(state)
        from centrex_tlf import hamiltonian
        until(self.end+self.context['physics']['post_decay_lifetimes']/float(hamiltonian.Γ))
        final=self.download()
        return np.stack((exit_rho,final),axis=1).astype(np.complex128),np.stack(populations,axis=1),dict(
            field_exit_wall_s_including_sampling=exit_wall,
            full_wall_s_including_sampling_and_post_decay=time.perf_counter()-started)
