__device__ const int H_ptr[]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338};
__device__ const int H_k[]={2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
__device__ const int D_ptr[]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20};
__device__ const int D_k[]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

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
C v0=phase_mod(real_arg(C(stage_t,0.),error),real_arg(C(3.7999999999999998,0),error),real_arg(C(10053096.491487337,0),error));
out[2*S+(S==1?0:b)]=v0;
}
__device__ void eval_variable(const C* params,int B,int b,double stage_t,double switch_t,C* out,int S,int* error){
out[1*S+(S==1?0:b)]=params[0*B+b];
}

extern "C" __global__ void coefficients(const C* params,double* clock,C* out,const C* table,long rows,int K,int B,int stage,double fraction,int skip,int* error){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B||(skip&&clock[3]!=0.))return;
 double stage_t=clock[0]+fraction*clock[1],switch_t=clock[0]+.5*clock[1];long row=(long)clock[4];
 if(b==0){clock[5]+=1.;if(1&&clock[1]>0.&&row<rows){out[0*B]=table[(row*7+stage)*K+0];out[2*B]=table[(row*7+stage)*K+2];}else eval_common(params,B,0,stage_t,switch_t,out,B,error);}
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
 z=lower?conj(bases[e]):bases[e];for(int p=H_ptr[e];p<H_ptr[e+1];p++){int k=H_k[p];C v=bases[k*E+e]*coeff[k*B+(varied[k]?b:0)];z+=lower?conj(v):v;}
 hr[x]=z.real();hi[x]=z.imag();
}
__global__ void build_diag(const C* coeff,const C* bases,const int* varied,double* diag,int n,int K,int B,
 const double* clock,int skip,int* error){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=(long)n*B||(skip&&clock[3]!=0.))return;
 int i=x/B,b=x%B;C z(0.,0.);z=bases[i];for(int p=D_ptr[i];p<D_ptr[i+1];p++){int k=D_k[p];z+=bases[k*n+i]*coeff[k*B+(varied[k]?b:0)];}
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
extern "C" __global__ void product_tile(const double* y,double* ar,double* ai,const int* ptr,const int* col,
 const double* hr,const double* hi,int B,int n,int HB,const double* clock,int skip){
 if(skip&&clock[3]!=0.)return;
 int j=blockIdx.y,b0=blockIdx.x*32;long P=(long)B*n*(n+1)/2;
 __shared__ double sr[40*32],si[40*32];
 for(int z=threadIdx.x;z<n*32;z+=blockDim.x){int k=z/32,b=b0+z%32;double r=0.,v=0.;
  if(b<B)readpk(y,k,j,b,B,n,P,r,v);sr[z]=r;si[z]=v;}
 __syncthreads();
 for(int z=threadIdx.x;z<n*32;z+=blockDim.x){int i=z/32,t=z%32,b=b0+t;if(b>=B)continue;
  double r=0.,v=0.;for(int p=ptr[i];p<ptr[i+1];p++){int k=col[p]*32+t;long h=(long)p*HB+(HB==1?0:b);
   r+=hr[h]*sr[k]-hi[h]*si[k];v+=hr[h]*si[k]+hi[h]*sr[k];}
  long out=((long)i*n+j)*B+b;ar[out]=r;ai[out]=v;
 }
}
