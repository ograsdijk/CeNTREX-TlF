__device__ __constant__ int GATHER[]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,1,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,2,41,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,3,42,80,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,4,43,81,118,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,5,44,82,119,155,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,6,45,83,120,156,191,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,7,46,84,121,157,192,226,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,8,47,85,122,158,193,227,260,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,9,48,86,123,159,194,228,261,293,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,10,49,87,124,160,195,229,262,294,325,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,11,50,88,125,161,196,230,263,295,326,356,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,12,51,89,126,162,197,231,264,296,327,357,386,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,13,52,90,127,163,198,232,265,297,328,358,387,415,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,14,53,91,128,164,199,233,266,298,329,359,388,416,443,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,15,54,92,129,165,200,234,267,299,330,360,389,417,444,470,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,16,55,93,130,166,201,235,268,300,331,361,390,418,445,471,496,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,17,56,94,131,167,202,236,269,301,332,362,391,419,446,472,497,521,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,18,57,95,132,168,203,237,270,302,333,363,392,420,447,473,498,522,545,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,19,58,96,133,169,204,238,271,303,334,364,393,421,448,474,499,523,546,568,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,20,59,97,134,170,205,239,272,304,335,365,394,422,449,475,500,524,547,569,590,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,21,60,98,135,171,206,240,273,305,336,366,395,423,450,476,501,525,548,570,591,611,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,22,61,99,136,172,207,241,274,306,337,367,396,424,451,477,502,526,549,571,592,612,631,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,23,62,100,137,173,208,242,275,307,338,368,397,425,452,478,503,527,550,572,593,613,632,650,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,24,63,101,138,174,209,243,276,308,339,369,398,426,453,479,504,528,551,573,594,614,633,651,668,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,25,64,102,139,175,210,244,277,309,340,370,399,427,454,480,505,529,552,574,595,615,634,652,669,685,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,26,65,103,140,176,211,245,278,310,341,371,400,428,455,481,506,530,553,575,596,616,635,653,670,686,701,715,716,717,718,719,720,721,722,723,724,725,726,727,728,27,66,104,141,177,212,246,279,311,342,372,401,429,456,482,507,531,554,576,597,617,636,654,671,687,702,716,729,730,731,732,733,734,735,736,737,738,739,740,741,28,67,105,142,178,213,247,280,312,343,373,402,430,457,483,508,532,555,577,598,618,637,655,672,688,703,717,730,742,743,744,745,746,747,748,749,750,751,752,753,29,68,106,143,179,214,248,281,313,344,374,403,431,458,484,509,533,556,578,599,619,638,656,673,689,704,718,731,743,754,755,756,757,758,759,760,761,762,763,764,30,69,107,144,180,215,249,282,314,345,375,404,432,459,485,510,534,557,579,600,620,639,657,674,690,705,719,732,744,755,765,766,767,768,769,770,771,772,773,774,31,70,108,145,181,216,250,283,315,346,376,405,433,460,486,511,535,558,580,601,621,640,658,675,691,706,720,733,745,756,766,775,776,777,778,779,780,781,782,783,32,71,109,146,182,217,251,284,316,347,377,406,434,461,487,512,536,559,581,602,622,641,659,676,692,707,721,734,746,757,767,776,784,785,786,787,788,789,790,791,33,72,110,147,183,218,252,285,317,348,378,407,435,462,488,513,537,560,582,603,623,642,660,677,693,708,722,735,747,758,768,777,785,792,793,794,795,796,797,798,34,73,111,148,184,219,253,286,318,349,379,408,436,463,489,514,538,561,583,604,624,643,661,678,694,709,723,736,748,759,769,778,786,793,799,800,801,802,803,804,35,74,112,149,185,220,254,287,319,350,380,409,437,464,490,515,539,562,584,605,625,644,662,679,695,710,724,737,749,760,770,779,787,794,800,805,806,807,808,809,36,75,113,150,186,221,255,288,320,351,381,410,438,465,491,516,540,563,585,606,626,645,663,680,696,711,725,738,750,761,771,780,788,795,801,806,810,811,812,813,37,76,114,151,187,222,256,289,321,352,382,411,439,466,492,517,541,564,586,607,627,646,664,681,697,712,726,739,751,762,772,781,789,796,802,807,811,814,815,816,38,77,115,152,188,223,257,290,322,353,383,412,440,467,493,518,542,565,587,608,628,647,665,682,698,713,727,740,752,763,773,782,790,797,803,808,812,815,817,818,39,78,116,153,189,224,258,291,323,354,384,413,441,468,494,519,543,566,588,609,629,648,666,683,699,714,728,741,753,764,774,783,791,798,804,809,813,816,818,819};

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
__device__ __forceinline__ long pk(int i,int j,int n){return GATHER[i*n+j];}
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
 if(skip&&clock[3]!=0.)return;
 const int T=32,R=16;
 int j=blockIdx.y,b0=blockIdx.x*T;long P=(long)B*n*(n+1)/2;
 extern __shared__ double sm[];double* sr=sm;double* si=sm+n*T;
 for(int z=threadIdx.x;z<n*T;z+=blockDim.x){int k=z/T,b=b0+z%T;double u=0.,v=0.;
  if(b<B)readpk(y,k,j,b,B,n,P,u,v);sr[z]=u;si[z]=v;}
 __syncthreads();
 for(int z=threadIdx.x;z<R*T;z+=blockDim.x){
  int i=blockIdx.z*R+z/T,b=b0+z%T,t=z%T;if(i>j||i>=n||b>=B)continue;
  double ar=0.,ai=0.,jr=0.,ji=0.;
  for(int k=ptr[i];k<ptr[i+1];k++){int a=col[k]*T+t;long h=(long)k*HB+(HB==1?0:b);
   ar+=hr[h]*sr[a]-hi[h]*si[a];ai+=hr[h]*si[a]+hi[h]*sr[a];}
  for(int k=ptr[j];k<ptr[j+1];k++){double u,v;readpk(y,col[k],i,b,B,n,P,u,v);long h=(long)k*HB+(HB==1?0:b);
   jr+=hr[h]*u-hi[h]*v;ji+=hr[h]*v+hi[h]*u;}
  long x=pk(i,j,n)*B+b;double u=y[x],v=y[P+x],d=diag[i*B+b]-diag[j*B+b],loss=-.5*(rates[i]+rates[j]);
  double r=ai+ji+d*v+loss*u,w=jr-ar-d*u+loss*v;
  if(i==j){for(int k=tp[i];k<tp[i+1];k++)r+=tr[k]*y[pk(tc[k],tc[k],n)*B+b];w=0.;}
  out[x]=r;out[P+x]=w;
 }
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