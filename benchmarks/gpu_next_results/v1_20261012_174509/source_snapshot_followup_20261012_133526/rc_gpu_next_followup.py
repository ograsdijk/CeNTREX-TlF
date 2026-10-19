"""Benchmark-only block/tiling, low-storage and waveform experiments."""
import math
from pathlib import Path
import numpy as np
import cupy as cp
from rc_gpu_storage_followup import PackedGPU
from rc_gpu_integrators import TS_C,TS_A

# Carpenter & Kennedy, NASA TM-109112 (1994), five-stage fourth-order 2N scheme.
CK_A=[0.,-567301805773/1357537059087,-2404267990393/2016746695238,
      -3550918686646/2091501179385,-1275806237668/842570457699]
CK_B=[1432997174477/9575080441755,5161836677717/13612068292357,
      1720146321549/2090206949498,3134564353537/4481467310338,2277821191437/14882151754819]
CK_C=[0.,1432997174477/9575080441755,2526269341429/6820363962896,
      2006345519317/3224310063776,2802321613138/2924317926251]

EXTRA=r'''
extern "C" {
__global__ void ls_update(double* y,double* r,const double* f,const double* clock,long D,double a,double b){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D||clock[1]<=0.)return;
 double v=(a==0.?0.:a*r[x])+clock[1]*f[x];r[x]=v;y[x]+=b*v;
}
__global__ void table_h(const double* clock,const double* table,const double* times,const double* params,int steps,int stage,double fraction,
 const int* positions,const double* br,const double* bi,double* hr,double* hi,int count,int n){
 int x=blockIdx.x*blockDim.x+threadIdx.x;
 __shared__ double coeff[10];
 if(threadIdx.x==0){
  // Tiny positive residual steps at output boundaries can increment the
  // accepted-step counter. Match physical time, not that counter.
  double t=clock[0],h=clock[1];int lo=0,hi=steps;
  while(lo<hi){int mid=(lo+hi)/2;if(times[2*mid]<t)lo=mid+1;else hi=mid;}
  int step=min(lo,steps-1);if(step>0 && fabs(times[2*(step-1)]-t)<fabs(times[2*step]-t))step--;
  bool match=h>1e-15 && fabs(times[2*step]-t)<1e-15 && fabs(times[2*step+1]-h)<1e-15;
  if(match){long base=((long)step*7+stage)*10;for(int c=0;c<10;c++)coeff[c]=table[base+c];}
  else{
   double u=t+fraction*h,z=(u-params[0])/params[1],env=exp(-.5*z*z);
   int q=((int)floor((t+.5*h)/2.5e-7+1e-8))%4;bool a=q<2,b=q==0||q==3;
   coeff[0]=(!a)*params[2]*env;coeff[1]=a*params[2]*env;
   coeff[2]=b*params[3]*env;coeff[3]=(!b)*params[3]*env;
   double phase=params[5]*sin(params[6]*u);coeff[4]=params[4]*cos(phase);
   for(int c=5;c<9;c++)coeff[c]=0.;coeff[9]=params[4]*sin(phase);
  }
 }
 __syncthreads();if(x>=count)return;
 int ij=positions[x],i=ij/n,j=ij%n;double r=0.,v=0.;
 for(int c=0;c<5;c++){double a=coeff[c],b=coeff[5+c]*(i<=j?1.:-1.);
 double u=br[c*n*n+ij],w=bi[c*n*n+ij];r+=a*u-b*w;v+=a*w+b*u;}
 hr[x]=r;hi[x]=v;
}
__global__ void unpack_density(const double* packed,double* output,int B,int n){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=(long)B*n*n)return;
 int b=x/(n*n),ij=x%(n*n),i=ij/n,j=ij%n;long P=(long)B*n*(n+1)/2;
 double r,v;readpk(packed,i,j,b,B,n,P,r,v);output[2*x]=r;output[2*x+1]=v;
}
__global__ void photon_rate(const double* y,double* values,const int* excited,int ne,int B,int n,int stage,double gamma){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B)return;double s=0.;
 for(int k=0;k<ne;k++){int i=excited[k];s+=y[((long)i*n-i*(i-1)/2)*B+b];}
 values[(long)stage*B+b]=gamma*s;
}
__global__ void photon_tsit(double* photons,const double* values,const double* weights,const double* clock,int B){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B||clock[1]<=0.)return;double s=0.;
 for(int k=0;k<6;k++)s+=weights[k]*values[(long)k*B+b];photons[b]+=clock[1]*s;
}
__global__ void photon_ls(double* photons,double* residual,const double* values,const double* clock,int B,int stage,double a,double b){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B||clock[1]<=0.)return;
 double r=(a==0.?0.:a*residual[i])+clock[1]*values[(long)stage*B+i];residual[i]=r;photons[i]+=b*r;
}
}
'''

def tile_source(n,tile):
    return r'''
extern "C" __global__ void product_tile(const double* y,double* ar,double* ai,const int* ptr,const int* col,
const double* hr,const double* hi,int B,int n,const double* clock,int skip){
 if(skip && clock[4]!=0.)return;
 const int T=TILE;int j=blockIdx.y,b0=blockIdx.x*T;long P=(long)B*n*(n+1)/2;
 __shared__ double sr[NLEVEL*TILE],si[NLEVEL*TILE];
 for(int z=threadIdx.x;z<n*T;z+=blockDim.x){int k=z/T,b=b0+z%T;double r=0.,v=0.;
 if(b<B)readpk(y,k,j,b,B,n,P,r,v);sr[z]=r;si[z]=v;}
 __syncthreads();
 for(int z=threadIdx.x;z<n*T;z+=blockDim.x){int i=z/T,t=z%T,b=b0+t;if(b>=B)continue;
 double r=0.,v=0.;for(int p=ptr[i];p<ptr[i+1];p++){
 int k=col[p]*T+t;double u=sr[k],w=si[k];r+=hr[p]*u-hi[p]*w;v+=hr[p]*w+hi[p]*u;}
 long out=((long)i*n+j)*B+b;ar[out]=r;ai[out]=v;}
}
'''.replace('NLEVEL',str(n)).replace('TILE',str(tile))

class NextGPU(PackedGPU):
    def __init__(self,*args,threads=256,tile=0,waveform=False,photons=False,excited=None,**kwargs):
        self.threads=threads;self.tile=tile;self.waveform=waveform;self.wave_table=None
        self.photons=photons;self._stage_counter=0
        self.low_storage=kwargs.get('integrator',args[5] if len(args)>5 else 'tsit5')=='ck54'
        super().__init__(*args,**kwargs)
        if tile and self.direct:raise ValueError('Shared column tile uses staged product')
        self.source+=EXTRA
        if tile:self.source+=tile_source(self.n,tile)
        module=cp.RawModule(code=self.source,options=('--std=c++14','-DSTATE=double'))
        for name in ['ls_update','table_h','unpack_density','photon_rate','photon_tsit','photon_ls']+(['product_tile'] if tile else []):
            self.kernels[name]=module.get_function(name)
        if self.low_storage:
            self.residual=cp.empty_like(self.y);self.derivative=cp.empty_like(self.y)
            self.ks=[self.derivative]*7
            del self.stages,self.temps,self.yn
        if waveform:self.prepare_waveform([self.end])
        if photons:
            if excited is None:raise ValueError('Photon integral needs excited-state indices')
            self.excited=cp.asarray(np.asarray(excited,np.int32))
            self.photon_values=cp.zeros((7,self.batch),cp.float64)
            self.photon_integral=cp.zeros(self.batch,cp.float64);self.photon_residual=cp.empty(self.batch,cp.float64)
            self.photon_weights=cp.asarray(TS_A[6,:6])

    def launch(self,name,args,count=None):
        if name=='product' and self.tile:
            self.kernels['product_tile'](((self.batch+self.tile-1)//self.tile,self.n),(self.threads,),args)
        else:
            count=self.size if count is None else count
            self.kernels[name](((count+self.threads-1)//self.threads,),(self.threads,),args)

    def prepare_waveform(self,targets):
        p=self.p.get();records=[];t=self.start
        fractions=CK_C if self.low_storage else TS_C
        for target in targets:
            while t<target-1e-15:
                factor=self.schedule_multiplier if self.scheduled and abs((t-self.schedule_center)/self.schedule_sigma)>.8 else 1.
                edge=(math.floor((t+1e-14)/2.5e-7)+1)*2.5e-7
                h=max(0.,min(self.dt*factor,edge-t,target-t))
                if h<=0.:raise RuntimeError('Waveform clock planner stalled')
                records.append((t,h));t+=h
        records=np.asarray(records);times=records[:,0,None]+records[:,1,None]*np.asarray(fractions)[None,:]
        quarter=(np.floor((records[:,0]+.5*records[:,1])/2.5e-7+1e-8).astype(int)%4)[:,None]
        aa=quarter<2;bb=(quarter==0)|(quarter==3)
        envelope=np.exp(-.5*((times-p[0])/p[1])**2)
        table=np.zeros((len(records),7,10))
        values=[(~aa)*p[2]*envelope,aa*p[2]*envelope,bb*p[3]*envelope,(~bb)*p[3]*envelope,
                p[4]*np.cos(p[5]*np.sin(p[6]*times))]
        for c,value in enumerate(values):table[:,:len(fractions),c]=value
        table[:,:len(fractions),9]=p[4]*np.sin(p[5]*np.sin(p[6]*times))
        # Field-exit c=1 belongs to the driven step; only following steps are dark.
        table[records[:,0]>=self.end-1e-15]=0.
        self.wave_table=cp.asarray(table);self.wave_times=cp.asarray(records);self.wave_steps=len(records)
        self.wave_clock_error=0.

    def rhs_new(self,y,out,fraction,skip=False,next_stage=None,row=0,current=0,build=True):
        stage=self._stage_counter%(5 if self.low_storage else 7)
        self._stage_counter+=1
        if self.photons and (self.low_storage or stage<6):
            from centrex_tlf import hamiltonian
            self.launch('photon_rate',(y,self.photon_values,self.excited,np.int32(self.excited.size),
                    np.int32(self.batch),np.int32(self.n),np.int32(stage),np.float64(hamiltonian.Γ)),self.batch)
        if build and self.waveform:
            fractions=CK_C if self.low_storage else TS_C
            wave_stage=min(range(len(fractions)),key=lambda k:abs(fractions[k]-fraction))
            self.launch('table_h',(self.clock,self.wave_table,self.wave_times,self.p,np.int32(self.wave_steps),np.int32(wave_stage),np.float64(fraction),
                  self.positions,self.br,self.bi,self.hr,self.hi,np.int32(self.positions.size),np.int32(self.n)),self.positions.size)
            build=False
        result=super().rhs_new(y,out,fraction,skip,next_stage,row,current,build)
        if self.photons and not self.low_storage and stage==6:
            self.launch('photon_tsit',(self.photon_integral,self.photon_values,self.photon_weights,self.clock,np.int32(self.batch)),self.batch)
        return result

    def block(self,steps=16):
        self._stage_counter=0
        if not self.low_storage:return super().block(steps)
        for _ in range(steps):
            self.kernels['begin_step']((1,),(1,),(self.clock,))
            for stage,(a,b,c) in enumerate(zip(CK_A,CK_B,CK_C)):
                self.rhs_new(self.y,self.derivative,c)
                self.launch('ls_update',(self.y,self.residual,self.derivative,self.clock,np.int64(2*self.size),np.float64(a),np.float64(b)),2*self.size)
                if self.photons:self.launch('photon_ls',(self.photon_integral,self.photon_residual,self.photon_values,self.clock,
                               np.int32(self.batch),np.int32(stage),np.float64(a),np.float64(b)),self.batch)
            self.kernels['control']((1,),(1,),(self.clock,self.errors,np.int32(self.batch),np.int32(0),np.int32(0)))

    def full_transit(self):
        if self.waveform and not getattr(self,'wave_full',False):
            from centrex_tlf import hamiltonian
            targets=list(np.linspace(self.start,self.end,201)[1:])
            targets.append(self.end+self.context['physics']['post_decay_lifetimes']/float(hamiltonian.Γ))
            self.prepare_waveform(targets)
            # Graphs capture array pointers, so refresh after waveform replacement.
            self.capture()
        return super().full_transit()

    def reset(self):
        super().reset()
        if self.photons and hasattr(self,'photon_integral'):self.photon_integral.fill(0.)

    def launch_to(self,start,target):
        super().launch_to(start,target)
        if self.photons and abs(target-self.end)<1e-14:self.photon_exit=self.photon_integral.copy()
