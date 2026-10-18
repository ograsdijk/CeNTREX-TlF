// Experimental RC GPU integration. STATE is double or float; arithmetic is double.
#ifndef STATE
#define STATE double
#endif
extern "C" {
__global__ void stage(STATE* dest,const STATE* base,const STATE* ks,const double* w,
                      const double* clock,long D,int row,int terms) {
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;
 double s=0.;for(int k=0;k<terms;k++) if(w[row*7+k]!=0.)s+=w[row*7+k]*(double)ks[k*D+x];
 dest[x]=(double)base[x]+clock[1]*s;
}
__global__ void start_fsal(STATE* ks,const double* clock,long D) {
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(x<D && clock[4]==1.)ks[x]=ks[6*D+x];
}
__global__ void product(const STATE* y,double* ar,double* ai,const int* ptr,const int* col,
 const double* hr,const double* hi,int B,int n,const double* clock,int skip) {
 long N=(long)B*n*n,x=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(x>=N || (skip && clock[4]!=0.))return;
 int b=x%B,ij=x/B,i=ij/n,j=ij%n;double r=0.,w=0.;
 // SPECIALIZED_PRODUCT
 for(int p=ptr[i];p<ptr[i+1];p++) {
  long k=((long)col[p]*n+j)*B+b;double u=y[k],v=y[N+k];
  r+=hr[p]*u-hi[p]*v;w+=hr[p]*v+hi[p]*u;
 }
 ar[x]=r;ai[x]=w;
}
__global__ void finish(const STATE* y,STATE* out,const double* ar,const double* ai,
 const double* diag,const double* mask,const double* det,const double* rates,
 const int* tp,const int* tc,const double* tr,int B,int n,const double* clock,int skip,
 STATE* next,const STATE* base,const STATE* ks,const double* weights,int row,int current,int fused) {
 long N=(long)B*n*n,x=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(x>=N)return;
 int b=x%B,ij=x/B,i=ij/n,j=ij%n;
 double r,w;
 if(skip && clock[4]!=0.){r=out[x];w=out[N+x];}
 else {
  long xt=((long)j*n+i)*B+b;double u=y[x],v=y[N+x];
  double d=diag[i]-diag[j]+det[b]*(mask[i]-mask[j]),loss=-.5*(rates[i]+rates[j]);
  r=ai[x]+ai[xt]+d*v+loss*u;w=ar[xt]-ar[x]-d*u+loss*v;
  if(i==j){for(int p=tp[i];p<tp[i+1];p++)r+=tr[p]*(double)y[((long)tc[p]*n+tc[p])*B+b];w=0.;}
  out[x]=r;out[N+x]=w;
 }
 if(fused){
  double sr=0.,si=0.;for(int k=0;k<=current;k++) {
   double a=weights[row*7+k];if(a!=0.){sr+=a*(k==current?r:(double)ks[k*2*N+x]);si+=a*(k==current?w:(double)ks[k*2*N+N+x]);}
  }
  next[x]=(double)base[x]+clock[1]*sr;next[N+x]=(double)base[N+x]+clock[1]*si;
 }
}
__global__ void errors(const STATE* y,const STATE* yn,const STATE* ks,const double* e,
 const double* clock,double* errors,int B,int n,double atol,double rtol) {
 __shared__ double sums[256];int b=blockIdx.x,t=threadIdx.x;long N=(long)B*n*n;
 double sum=0.;for(int ij=t;ij<n*n;ij+=blockDim.x){long x=(long)ij*B+b;double r=0.,v=0.;
  for(int k=0;k<7;k++){r+=e[k]*(double)ks[k*2*N+x];v+=e[k]*(double)ks[k*2*N+N+x];}
  double a=hypot((double)y[x],(double)y[N+x]),c=hypot((double)yn[x],(double)yn[N+x]);
  double s=atol+rtol*fmax(a,c);r*=clock[1]/s;v*=clock[1]/s;sum+=r*r+v*v;
 }sums[t]=sum;__syncthreads();for(int d=128;d>0;d/=2){if(t<d)sums[t]+=sums[t+d];__syncthreads();}
 if(t==0)errors[b]=sqrt(sums[0]/(n*n));
}
__global__ void control(double* clock,const double* errors,int B,int adaptive,int tsit) {
 double h=clock[1];if(h<=0.){clock[5]=0.;return;}
 double err=0.;if(adaptive)for(int b=0;b<B;b++)err=fmax(err,isfinite(errors[b])?errors[b]:1e300);
 int accept=!adaptive || err<=1.;clock[5]=accept;
 if(accept){clock[0]+=h;clock[6]+=1.;
  double q=clock[0]/2.5e-7;clock[4]=(tsit && fabs(q-nearbyint(q))>1e-7)?1.:0.;
 }else{clock[7]+=1.;clock[4]=2.;}
 if(adaptive){double fac=err==0.?5.:fmin(5.,fmax(.2,.9*pow(err,-.2)));
  if(!accept)fac=fmin(1.,fac);clock[3]=fmin(2.5e-7,fmax(1e-14,h*fac));}
}
__global__ void accept_step(STATE* y,const STATE* yn,const double* clock,long D){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x<D && clock[5]!=0.)y[x]=yn[x];
}
}
