# Frozen pre-FMA generic CUDA source for benchmark controls only.
"""FP64 sparse commutator and fixed-step kernels (compiled lazily by CuPy)."""

HEADER = r"""
#include <cupy/complex.cuh>
typedef complex<double> C;
const double PI=3.141592653589793238462643383279502884;
__device__ double real_arg(C z,int* error){if(fabs(z.imag())>1e-12)atomicExch(error,2);return z.real();}
__device__ double frac(double x){return x-floor(x);}
__device__ double g1(double x,double m,double s){double z=(x-m)/s;return exp(-.5*z*z);}
__device__ double g2(double x,double y,double a,double mx,double my,double sx,double sy){return a*g1(x,mx,sx)*g1(y,my,sy);}
__device__ double g2rot(double x,double y,double amp,double mx,double my,double sx,double sy,double theta){
 double c=cos(theta),s=sin(theta),dx=x-mx,dy=y-my;
 double a=c*c/(2.*sx*sx)+s*s/(2.*sy*sy),b=sin(2.*theta)/(2.*sx*sx)-sin(2.*theta)/(2.*sy*sy);
 double d=s*s/(2.*sx*sx)+c*c/(2.*sy*sy);return amp*exp(-(a*dx*dx+b*dx*dy+d*dy*dy));
}
__device__ C phase_mod(double t,double beta,double omega){double p=beta*sin(omega*t);return C(cos(p),sin(p));}
__device__ C pol_mod(double t,double gamma,double omega){double p=.5*gamma*sin(omega*t),a=.5*(cos(p)+sin(p));return C(a,a);}
__device__ double gate_duty(double t,double duty,double invT,double phase){double f=frac(t*invT+phase/(2.*PI));if(f<=0.)f+=1.;return f<duty?1.:0.;}
__device__ __forceinline__ long pk(int i,int j,int n){if(i>j){int t=i;i=j;j=t;}return (long)i*n-i*(i+1)/2+j;}
__device__ __forceinline__ long density_pairs(int n){
#ifdef REDUCED_PAIRS
 return REDUCED_PAIRS;
#else
 return (long)n*(n+1)/2;
#endif
}
__device__ __forceinline__ long density_pk(int i,int j,int n){
#ifdef REDUCED_PAIRS
 return density_index[pk(i,j,n)];
#else
 return pk(i,j,n);
#endif
}
__device__ __forceinline__ double population(const double* y,int i,int b,int B,int n){
 long q=density_pk(i,i,n);return q<0?0.:y[q*B+b];
}
__device__ __forceinline__ void readpk(const double* y,int i,int j,int b,int B,int n,long P,double& r,double& v){
#ifdef REDUCED_PAIRS
 long q=density_pk(i,j,n);if(q<0){r=0.;v=0.;return;}long x=q*B+b;r=y[x];v=(i>j?-1.:1.)*y[P+x];
#else
 long x=pk(i,j,n)*B+b;r=y[x];v=(i>j?-1.:1.)*y[P+x];
#endif
}
"""

KERNELS = r"""
extern "C" {
// clock: current time, step size, segment step limit, FSAL valid, step index.
__global__ void set_limit(double* clock,long limit){clock[2]=(double)limit;}
__global__ void begin_step(double* clock,const double* grid,const int* resets,long steps){
 long i=(long)clock[4];if(i>=(long)clock[2]||i>=steps){clock[1]=0.;clock[3]=0.;return;}
 clock[0]=grid[2*i];clock[1]=grid[2*i+1];if(resets[i])clock[3]=0.;
}
__global__ void end_step(double* clock,int fsal){if(clock[1]>0.){clock[0]+=clock[1];clock[4]+=1.;clock[3]=fsal;}}
__global__ void build_h(const C* coeff,const C* bases,const int* varied,const int* rows,const int* cols,
 double* hr,double* hi,int E,int K,int B,int HB,const double* clock,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=(long)E*HB||(skip&&clock[3]!=0.))return;
 int e=x/HB,b=x%HB;C z(0.,0.);bool lower=rows[e]>cols[e];
 for(int k=0;k<K;k++){C v=bases[k*E+e]*coeff[k*B+(varied[k]?b:0)];z+=lower?conj(v):v;}
 hr[x]=z.real();hi[x]=z.imag();
}
__global__ void build_diag(const C* coeff,const C* bases,const int* varied,double* diag,int n,int K,int B,
 const double* clock,int skip,int* error){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=(long)n*B||(skip&&clock[3]!=0.))return;
 int i=x/B,b=x%B;C z(0.,0.);for(int k=0;k<K;k++)z+=bases[k*n+i]*coeff[k*B+(varied[k]?b:0)];
 if(fabs(z.imag())>1e-10)atomicExch(error,3);diag[x]=z.real();
}
__global__ void product(const double* y,double* ar,double* ai,const int* ptr,const int* col,
 const double* hr,const double* hi,int B,int n,int HB,const double* clock,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,P=(long)B*density_pairs(n);
 if(x>=(long)B*n*n||(skip&&clock[3]!=0.))return;
 int b=x%B,ij=x/B,i=ij/n,j=ij%n;double r=0.,v=0.;
 for(int p=ptr[i];p<ptr[i+1];p++){double u,w;readpk(y,col[p],j,b,B,n,P,u,w);long h=(long)p*HB+(HB==1?0:b);r+=hr[h]*u-hi[h]*w;v+=hr[h]*w+hi[h]*u;}
 ar[x]=r;ai[x]=v;
}
__global__ void finish(const double* y,double* out,const double* ar,const double* ai,const double* diag,
 const double* rates,const int* tp,const int* tc,const double* tr,const int* rows,const int* cols,
 int B,int n,const double* clock,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,P=(long)B*density_pairs(n);
 if(x>=P||(skip&&clock[3]!=0.))return;
 int b=x%B,p=x/B,i=rows[p],j=cols[p];long a=((long)i*n+j)*B+b,at=((long)j*n+i)*B+b;
 double u=y[x],v=y[P+x],d=diag[i*B+b]-diag[j*B+b],loss=-.5*(rates[i]+rates[j]);
 double r=ai[a]+ai[at]+d*v+loss*u,w=ar[at]-ar[a]-d*u+loss*v;
 if(i==j){for(int k=tp[i];k<tp[i+1];k++)r+=tr[k]*population(y,tc[k],b,B,n);w=0.;}
 out[x]=r;out[P+x]=w;
}
__global__ void direct_rhs(const double* y,double* out,const int* ptr,const int* col,const double* hr,const double* hi,
 const double* diag,const double* rates,const int* tp,const int* tc,const double* tr,const int* rows,const int* cols,
 int B,int n,int HB,const double* clock,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,P=(long)B*density_pairs(n);
 if(x>=P||(skip&&clock[3]!=0.))return;
 int b=x%B,p=x/B,i=rows[p],j=cols[p];double ar=0.,ai=0.,jr=0.,ji=0.;
 for(int k=ptr[i];k<ptr[i+1];k++){double u,v;readpk(y,col[k],j,b,B,n,P,u,v);long h=(long)k*HB+(HB==1?0:b);ar+=hr[h]*u-hi[h]*v;ai+=hr[h]*v+hi[h]*u;}
 for(int k=ptr[j];k<ptr[j+1];k++){double u,v;readpk(y,col[k],i,b,B,n,P,u,v);long h=(long)k*HB+(HB==1?0:b);jr+=hr[h]*u-hi[h]*v;ji+=hr[h]*v+hi[h]*u;}
 double u=y[x],v=y[P+x],d=diag[i*B+b]-diag[j*B+b],loss=-.5*(rates[i]+rates[j]);
 double r=ai+ji+d*v+loss*u,w=jr-ar-d*u+loss*v;
 if(i==j){for(int k=tp[i];k<tp[i+1];k++)r+=tr[k]*population(y,tc[k],b,B,n);w=0.;}
 out[x]=r;out[P+x]=w;
}
__global__ void integral_rhs(const double* y,double* out,const double* weights,int B,int n,const double* clock,int skip){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B||(skip&&clock[3]!=0.))return;
 double r=0.;for(int i=0;i<n;i++)r+=weights[i]*population(y,i,b,B,n);out[2*(long)B*density_pairs(n)+b]=r;
}
__global__ void ls_update(double* y,double* residual,const double* derivative,const double* clock,long D,double a,double b){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;
 double r=(a==0.?0.:a*residual[x])+clock[1]*derivative[x];residual[x]=r;y[x]+=b*r;
}
// Public packed layout is diagonals followed by interleaved Re/Im upper entries.
__global__ void sample(const double* y,double* output,const int* public_pk,const int* public_im,
 const int* selected_i,const int* selected_j,const double* weights,int B,int n,int width,int mode,int sample){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,P=(long)B*density_pairs(n);if(x>=(long)B*width)return;
 int b=x/width,k=x%width;double v=0.;
 if(mode==0||mode==5){if(mode==5&&k==n*n)v=y[2*P+b];else if(public_pk[k]>=0)v=y[(long)public_pk[k]*B+b+(public_im[k]?P:0)];}
 if(mode==1)v=population(y,k,b,B,n);
 if(mode==2){int i=selected_i[k/2],j=selected_j[k/2];long q=density_pk(i,j,n),a=q*B+b;
  if(q>=0)v=k%2?(i==j?0.:(i>j?-1.:1.)*y[P+a]):y[a];}
 if(mode==3)v=y[2*P+b];
 if(mode==4)for(int i=0;i<n;i++)v+=weights[i]*population(y,i,b,B,n);
 output[(long)sample*B*width+x]=v;
}
}
"""


COMPONENT_KERNELS = r"""
extern "C" __global__ void build_components(const C* coeff,const C* bases,const int* rows,const int* cols,const int* groups,const long long* offsets,
 double* hr,double* hi,int E,int K,const double* clock,int skip){
 int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=E||(skip&&clock[3]!=0.))return;
#if HYBRID_H_COMPONENTS
 if(groups[e]<0)return;
#endif
 C z(0.,0.);for(int k=0;k<K;k++)z+=bases[k*E+e]*coeff[k];if(rows[e]>cols[e])z=conj(z);
 long h=e;
#if HYBRID_H_COMPONENTS
 h=offsets[e];
#endif
 hr[h]=z.real();hi[h]=z.imag();
}
extern "C" __global__ void build_remainder(const C* coeff,const C* bases,const int* varied,const int* rows,const int* cols,
 const int* remainder,const long long* offsets,double* hr,double* hi,int R,int E,int K,int B,const double* clock,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=(long)R*B||(skip&&clock[3]!=0.))return;
 int e=remainder[x/B],b=x%B;C z(0.,0.);for(int k=0;k<K;k++)z+=bases[k*E+e]*coeff[k*B+(varied[k]?b:0)];
 if(rows[e]>cols[e])z=conj(z);long h=offsets[e]+b;hr[h]=z.real();hi[h]=z.imag();
}
__device__ __forceinline__ void component_h(int k,int b,int B,const double* hr,const double* hi,const int* groups,const long long* offsets,const double* amplitude,double& r,double& w){
 long h=k;
#if HYBRID_H_COMPONENTS
 h=offsets[k]+(groups[k]<0?b:0);
#endif
 r=hr[h];w=hi[h];
#if !SINGLE_H_COMPONENT
 double a=groups[k]<0?1.:amplitude[groups[k]*B+b];r*=a;w*=a;
#endif
}
extern "C" __global__ void component_rhs(const double* y,double* out,const int* ptr,const int* col,
 const double* hr,const double* hi,const int* groups,const long long* offsets,const double* amplitude,
 const double* diag,const double* rates,const int* tp,const int* tc,const double* tr,const int* rows,const int* cols,
 int B,int n,const double* clock,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,P=(long)B*density_pairs(n);
 if(x>=P||(skip&&clock[3]!=0.))return;
 int b=x%B,p=x/B,i=rows[p],j=cols[p];double ar=0.,ai=0.,jr=0.,ji=0.;
 for(int k=ptr[i];k<ptr[i+1];k++){double u,v;readpk(y,col[k],j,b,B,n,P,u,v);
  double r,w;component_h(k,b,B,hr,hi,groups,offsets,amplitude,r,w);
  ar+=r*u-w*v;ai+=r*v+w*u;}
 for(int k=ptr[j];k<ptr[j+1];k++){double u,v;readpk(y,col[k],i,b,B,n,P,u,v);
  double r,w;component_h(k,b,B,hr,hi,groups,offsets,amplitude,r,w);
  jr+=r*u-w*v;ji+=r*v+w*u;}
 double scale=1.;
#if SINGLE_H_COMPONENT
 scale=amplitude[b];
#endif
 double u=y[x],v=y[P+x],d=diag[i*B+b]-diag[j*B+b],loss=-.5*(rates[i]+rates[j]);
 double r=scale*(ai+ji)+d*v+loss*u,w=scale*(jr-ar)-d*u+loss*v;
 if(i==j){for(int k=tp[i];k<tp[i+1];k++)r+=tr[k]*population(y,tc[k],b,B,n);w=0.;}
 out[x]=r;out[P+x]=w;
}
"""


def tile_kernel(n: int, tile: int) -> str:
    return r"""
extern "C" __global__ void product_tile(const double* y,double* ar,double* ai,const int* ptr,const int* col,
 const double* hr,const double* hi,int B,int n,int HB,const double* clock,int skip){
 if(skip&&clock[3]!=0.)return;
 int j=blockIdx.y,b0=blockIdx.x*TILE;long P=(long)B*density_pairs(n);
 __shared__ double sr[NLEVEL*TILE],si[NLEVEL*TILE];
 for(int z=threadIdx.x;z<n*TILE;z+=blockDim.x){int k=z/TILE,b=b0+z%TILE;double r=0.,v=0.;
  if(b<B)readpk(y,k,j,b,B,n,P,r,v);sr[z]=r;si[z]=v;}
 __syncthreads();
 for(int z=threadIdx.x;z<n*TILE;z+=blockDim.x){int i=z/TILE,t=z%TILE,b=b0+t;if(b>=B)continue;
  double r=0.,v=0.;for(int p=ptr[i];p<ptr[i+1];p++){int k=col[p]*TILE+t;long h=(long)p*HB+(HB==1?0:b);
   r+=hr[h]*sr[k]-hi[h]*si[k];v+=hr[h]*si[k]+hi[h]*sr[k];}
  long out=((long)i*n+j)*B+b;ar[out]=r;ai[out]=v;
 }
}
""".replace("NLEVEL", str(n)).replace("TILE", str(tile))
