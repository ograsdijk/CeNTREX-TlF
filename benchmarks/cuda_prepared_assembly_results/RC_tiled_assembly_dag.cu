__device__ const int H_ptr[]={0,2,4,6,8,9,11,12,13,14,16,18,20,22,24,26,27,29,30,31,33,34,36,38,40,42,44,45,47,48,50,52,54,56,58,60,61,63,64,66,68,70,72,74,76,78,80,82,83,85,87,88,90,92,94,96,98,100,102,104,105,107,109,111,113,115,117,119,121,123,125,126,128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,166,168,170,172,174,175,177,179,180,182,184,186,188,190,192,194,196,198,200,202,204,206,208,210,212,214,216,218,220,222,224,226,228,230,232,234,236,238,240,242,244,246,248,250,251,253,255,256,257,259,261,262,264,266,267,269,271,273,275,277,279,281,283,284,286,287,288,290,292,294,296,298,300,302,304,306,308,310,312,313,315,317,318,319,321,323,325,327,329,331,333,335,337,339,340,342,343,345,347,349,351,353,355,357,359,361,363,365,367,369,371,373,374,376,377,378,379,381,383,385,387,389,391,393,394,396,398,399,401,403,405,407,409,410,412,413,414,415,417,419,421,423,425,427,429,431,433,435,437,439,441,443,445,446,448,449,450,452,454,456,458,460,462,464,466,468,470,471,473,474,476,478,480,482,484,486,488,490,492,494,496,498,500,501,503,504,505,506,508,510,512,514,516,518,520,522,524,526,528,530,532,534,536,538,539,541,542,543,544,546,547,549,551,553,555,557,558,560,562,563,565,567,569,571,573,575,577,579,580,582,583,584,585,586,588,589,591,593,595,597,599,601,603,605,607,609,611,613,615,617,619,620,622,623,624,626,628,630,632,634,636,638,640,642,644,646,648,650,651,653,654,656,658,660,662,664,666,668,670,672,673,675,677,679,681,683,685,687,689,691,693,695,697,699,701,702,703,704,706,708,710,712,714,716,718,720,722,724,726,728,730,732,734,736,737,738,740,741,743,745,747,749,751,753,755,757,758,760,762,763,765,767,769,771,772,773,774,776,777,779,781,783,785,787,789,791,793,795,797,799,801,803,805,807,808,810,812,814,816,818,820,822,824,826,828,830,832,834,836,837,839,841,843,845,847,849,851,853,855,857,859,861,863,865,867,869,871,873,875,877,879,881,883,885,887,889,891,893,895,897,899,901,902,904,906,907,909,911,913,915,917,919,921,923,925,927,929,931,933,935,937,939,941,943,945,947,949,951,953,955,957,959,961,963,965,967,969,971,973,975,977,978,980,982,984,986,988,990,992,994,996,998,1000,1001,1003,1005,1007,1009,1011,1013,1015,1017,1018,1020,1022,1023,1025,1027,1029,1031,1033,1035,1037,1039,1041,1043,1045,1047,1049,1051,1053,1055,1057,1059,1061,1063,1065,1067,1069,1071,1073,1075,1076,1078,1080,1082,1084,1086,1088,1090,1091,1093,1095,1097,1099,1101,1103,1105,1107,1108,1110,1112,1114,1116,1118,1119,1121,1123,1124,1126,1128,1130,1132,1134,1136,1138,1140,1142,1143,1145,1147,1149,1151,1153,1155,1157,1159,1161,1163,1165,1167,1169,1170,1172,1174,1176,1178,1180,1181,1183,1185,1187,1189,1191,1192,1194,1196,1198,1200,1202,1203,1205,1207,1209,1211,1213,1214,1216,1218,1220,1222,1224,1226,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258};
__device__ const int H_k[]={3,4,3,4,3,4,3,4,3,3,4,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,3,4,4,3,3,4,3,3,4,3,4,3,4,3,4,3,4,3,3,4,3,3,4,3,4,3,4,3,4,3,4,3,4,3,3,4,3,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,3,4,3,4,3,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,3,4,3,4,3,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,6,5,6,5,6,7,7,3,4,3,4,3,3,4,3,4,3,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,6,5,6,7,7,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,6,5,6,5,6,7,7,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,7,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,7,7,3,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,3,4,5,6,6,5,6,5,6,5,6,5,6,5,6,6,5,6,7,7,7,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,7,7,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,7,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,7,4,3,3,4,3,4,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,7,7,3,3,4,3,3,4,3,4,3,4,3,4,3,4,3,3,4,5,6,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,7,7,7,4,3,4,3,3,4,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,7,7,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,7,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,7,3,3,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,7,3,3,4,3,3,4,3,4,3,4,3,4,5,6,5,6,5,6,5,6,6,5,6,5,6,6,5,6,5,6,5,6,5,6,7,7,3,3,4,3,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,7,3,4,3,4,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,7,3,4,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7};
__device__ const int D_ptr[]={0,0,0,0,0,2,4,6,8,10,12,14,16,18,20,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,100,100,100,100,100,100};
__device__ const int D_k[]={1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8};

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
C v8=(params[14*B+b]*v3);
C v9=(v7*v8);
out[5*S+(S==1?0:b)]=v9;
C v10=(C(-1,0)*v7);
C v11=(C(1,0)+v10);
C v12=(v11*v8);
out[6*S+(S==1?0:b)]=v12;
C v13=phase_mod(real_arg(C(stage_t,0.),error),real_arg(params[1*B+b],error),real_arg(params[2*B+b],error));
C v14=(params[0*B+b]*v13);
C v15=(params[4*B+b]*v14);
out[7*S+(S==1?0:b)]=v15;
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
 __shared__ double sr[70*32],si[70*32];
 for(int z=threadIdx.x;z<n*32;z+=blockDim.x){int k=z/32,b=b0+z%32;double r=0.,v=0.;
  if(b<B)readpk(y,k,j,b,B,n,P,r,v);sr[z]=r;si[z]=v;}
 __syncthreads();
 for(int z=threadIdx.x;z<n*32;z+=blockDim.x){int i=z/32,t=z%32,b=b0+t;if(b>=B)continue;
  double r=0.,v=0.;for(int p=ptr[i];p<ptr[i+1];p++){int k=col[p]*32+t;long h=(long)p*HB+(HB==1?0:b);
   r+=hr[h]*sr[k]-hi[h]*si[k];v+=hr[h]*si[k]+hi[h]*sr[k];}
  long out=((long)i*n+j)*B+b;ar[out]=r;ai[out]=v;
 }
}
