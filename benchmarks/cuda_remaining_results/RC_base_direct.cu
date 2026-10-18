
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
__device__ __forceinline__ void readpk(const double* y,int i,int j,int b,int B,int n,long P,double& r,double& v){
 long x=pk(i,j,n)*B+b;r=y[x];v=(i>j?-1.:1.)*y[P+x];
}
__device__ void eval_common(const C* params,int B,int b,double stage_t,double switch_t,C* out,int S,int* error){
out[0*S+(S==1?0:b)]=C(1,0);
out[2*S+(S==1?0:b)]=params[5*B+b];
C v0=C(sin(real_arg(params[6*B+b],error)*real_arg(C(switch_t,0.),error)+real_arg(params[7*B+b],error))>=0.?1.:0.,0.);
C v1=(C(-1,0)*v0);
C v2=(C(1,0)+v1);
C v3=C(g1(real_arg(C(stage_t,0.),error),real_arg(params[11*B+b],error),real_arg(params[12*B+b],error)),0.);
C v4=(params[13*B+b]*v3);
C v5=(v2*v4);
out[3*S+(S==1?0:b)]=v5;
C v6=(v0*v4);
out[4*S+(S==1?0:b)]=v6;
C v7=C(sin(real_arg(params[9*B+b],error)*real_arg(C(switch_t,0.),error)+real_arg(params[10*B+b],error))>=0.?1.:0.,0.);
C v8=C(g1(real_arg(C(stage_t,0.),error),real_arg(params[11*B+b],error),real_arg(params[12*B+b],error)),0.);
C v9=(params[14*B+b]*v8);
C v10=(v7*v9);
out[5*S+(S==1?0:b)]=v10;
C v11=(C(-1,0)*v7);
C v12=(C(1,0)+v11);
C v13=(v12*v9);
out[6*S+(S==1?0:b)]=v13;
C v14=phase_mod(real_arg(C(stage_t,0.),error),real_arg(params[1*B+b],error),real_arg(params[2*B+b],error));
C v15=(params[0*B+b]*v14);
C v16=(params[4*B+b]*v15);
out[7*S+(S==1?0:b)]=v16;
out[8*S+(S==1?0:b)]=params[8*B+b];
}
__device__ void eval_variable(const C* params,int B,int b,double stage_t,double switch_t,C* out,int S,int* error){
out[1*S+(S==1?0:b)]=params[3*B+b];
}

extern "C" __global__ void coefficients(const C* params,double* clock,C* out,const C* table,long rows,int K,int B,int stage,double fraction,int skip,int* error){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B||(skip&&clock[3]!=0.))return;
 double stage_t=clock[0]+fraction*clock[1],switch_t=clock[0]+.5*clock[1];long row=(long)clock[4];
 if(b==0){clock[5]+=1.;if(1&&clock[1]>0.&&row<rows){out[0*B]=table[(row*7+stage)*K+0];out[2*B]=table[(row*7+stage)*K+2];out[3*B]=table[(row*7+stage)*K+3];out[4*B]=table[(row*7+stage)*K+4];out[5*B]=table[(row*7+stage)*K+5];out[6*B]=table[(row*7+stage)*K+6];out[7*B]=table[(row*7+stage)*K+7];out[8*B]=table[(row*7+stage)*K+8];}else eval_common(params,B,0,stage_t,switch_t,out,B,error);}
 eval_variable(params,B,b,stage_t,switch_t,out,B,error);
}
extern "C" __global__ void precompute(const C* params,const double* grid,C* table,long rows,int K,int B,int* error){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=rows*7)return;long row=x/7;int stage=x%7;
 const double fractions[7]={0,0.161,0.32700000000000001,0.90000000000000002,0.98002554090450966,1,1};double h=grid[2*row+1],t=grid[2*row];
 eval_common(params,B,0,t+fractions[stage]*h,t+.5*h,table+x*K,1,error);
}
extern "C" __global__ void validate(const double* y,long D,int* error){long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x<D&&!isfinite(y[x]))atomicExch(error,1);}

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
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,P=(long)B*n*(n+1)/2;
 if(x>=(long)B*n*n||(skip&&clock[3]!=0.))return;
 int b=x%B,ij=x/B,i=ij/n,j=ij%n;double r=0.,v=0.;
 for(int p=ptr[i];p<ptr[i+1];p++){double u,w;readpk(y,col[p],j,b,B,n,P,u,w);long h=(long)p*HB+(HB==1?0:b);r+=hr[h]*u-hi[h]*w;v+=hr[h]*w+hi[h]*u;}
 ar[x]=r;ai[x]=v;
}
__global__ void finish(const double* y,double* out,const double* ar,const double* ai,const double* diag,
 const double* rates,const int* tp,const int* tc,const double* tr,const int* rows,const int* cols,
 int B,int n,const double* clock,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,P=(long)B*n*(n+1)/2;
 if(x>=P||(skip&&clock[3]!=0.))return;
 int b=x%B,p=x/B,i=rows[p],j=cols[p];long a=((long)i*n+j)*B+b,at=((long)j*n+i)*B+b;
 double u=y[x],v=y[P+x],d=diag[i*B+b]-diag[j*B+b],loss=-.5*(rates[i]+rates[j]);
 double r=ai[a]+ai[at]+d*v+loss*u,w=ar[at]-ar[a]-d*u+loss*v;
 if(i==j){for(int k=tp[i];k<tp[i+1];k++)r+=tr[k]*y[pk(tc[k],tc[k],n)*B+b];w=0.;}
 out[x]=r;out[P+x]=w;
}
__global__ void direct_rhs(const double* y,double* out,const int* ptr,const int* col,const double* hr,const double* hi,
 const double* diag,const double* rates,const int* tp,const int* tc,const double* tr,const int* rows,const int* cols,
 int B,int n,int HB,const double* clock,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,P=(long)B*n*(n+1)/2;
 if(x>=P||(skip&&clock[3]!=0.))return;
 int b=x%B,p=x/B,i=rows[p],j=cols[p];double ar=0.,ai=0.,jr=0.,ji=0.;
 for(int k=ptr[i];k<ptr[i+1];k++){double u,v;readpk(y,col[k],j,b,B,n,P,u,v);long h=(long)k*HB+(HB==1?0:b);ar+=hr[h]*u-hi[h]*v;ai+=hr[h]*v+hi[h]*u;}
 for(int k=ptr[j];k<ptr[j+1];k++){double u,v;readpk(y,col[k],i,b,B,n,P,u,v);long h=(long)k*HB+(HB==1?0:b);jr+=hr[h]*u-hi[h]*v;ji+=hr[h]*v+hi[h]*u;}
 double u=y[x],v=y[P+x],d=diag[i*B+b]-diag[j*B+b],loss=-.5*(rates[i]+rates[j]);
 double r=ai+ji+d*v+loss*u,w=jr-ar-d*u+loss*v;
 if(i==j){for(int k=tp[i];k<tp[i+1];k++)r+=tr[k]*y[pk(tc[k],tc[k],n)*B+b];w=0.;}
 out[x]=r;out[P+x]=w;
}
__global__ void integral_rhs(const double* y,double* out,const double* weights,int B,int n,const double* clock,int skip){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B||(skip&&clock[3]!=0.))return;
 double r=0.;for(int i=0;i<n;i++)r+=weights[i]*y[pk(i,i,n)*B+b];out[(long)B*n*(n+1)+b]=r;
}
__global__ void ls_update(double* y,double* residual,const double* derivative,const double* clock,long D,double a,double b){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;
 double r=(a==0.?0.:a*residual[x])+clock[1]*derivative[x];residual[x]=r;y[x]+=b*r;
}
// Public packed layout is diagonals followed by interleaved Re/Im upper entries.
__global__ void sample(const double* y,double* output,const int* public_pk,const int* public_im,
 const int* selected_i,const int* selected_j,const double* weights,int B,int n,int width,int mode,int sample){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,P=(long)B*n*(n+1)/2;if(x>=(long)B*width)return;
 int b=x/width,k=x%width;double v=0.;
 if(mode==0||mode==5){if(mode==5&&k==n*n)v=y[2*P+b];else v=y[(long)public_pk[k]*B+b+(public_im[k]?P:0)];}
 if(mode==1)v=y[pk(k,k,n)*B+b];
 if(mode==2){int i=selected_i[k/2],j=selected_j[k/2];long a=pk(i,j,n)*B+b;
  v=k%2?(i==j?0.:(i>j?-1.:1.)*y[P+a]):y[a];}
 if(mode==3)v=y[2*P+b];
 if(mode==4)for(int i=0;i<n;i++)v+=weights[i]*y[pk(i,i,n)*B+b];
 output[(long)sample*B*width+x]=v;
}
}
extern "C" __global__ void stage_1(double* out,const double* y,const double* k0,const double* k1,const double* k2,const double* k3,const double* k4,const double* k5,const double* k6,const double* clock,long D){long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x<D)out[x]=y[x]+clock[1]*((0.161)*k0[x]);}
extern "C" __global__ void stage_2(double* out,const double* y,const double* k0,const double* k1,const double* k2,const double* k3,const double* k4,const double* k5,const double* k6,const double* clock,long D){long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x<D)out[x]=y[x]+clock[1]*((-0.0084806554923569887)*k0[x]+(0.335480655492357)*k1[x]);}
extern "C" __global__ void stage_3(double* out,const double* y,const double* k0,const double* k1,const double* k2,const double* k3,const double* k4,const double* k5,const double* k6,const double* clock,long D){long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x<D)out[x]=y[x]+clock[1]*((2.8971530571054935)*k0[x]+(-6.3594484899750752)*k1[x]+(4.3622954328695815)*k2[x]);}
extern "C" __global__ void stage_4(double* out,const double* y,const double* k0,const double* k1,const double* k2,const double* k3,const double* k4,const double* k5,const double* k6,const double* clock,long D){long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x<D)out[x]=y[x]+clock[1]*((5.3258648284392569)*k0[x]+(-11.748883564062828)*k1[x]+(7.4955393428898365)*k2[x]+(-0.092495066361755252)*k3[x]);}
extern "C" __global__ void stage_5(double* out,const double* y,const double* k0,const double* k1,const double* k2,const double* k3,const double* k4,const double* k5,const double* k6,const double* clock,long D){long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x<D)out[x]=y[x]+clock[1]*((5.8614554429464203)*k0[x]+(-12.92096931784711)*k1[x]+(8.1593678985761589)*k2[x]+(-0.071584973281400996)*k3[x]+(-0.028269050394068383)*k4[x]);}
extern "C" __global__ void stage_6(double* out,const double* y,const double* k0,const double* k1,const double* k2,const double* k3,const double* k4,const double* k5,const double* k6,const double* clock,long D){long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x<D)out[x]=y[x]+clock[1]*((0.09646076681806523)*k0[x]+(0.01)*k1[x]+(0.4798896504144996)*k2[x]+(1.3790085741037419)*k3[x]+(-3.2900695154360808)*k4[x]+(2.3247105240997739)*k5[x]);}