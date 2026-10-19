// Experimental double-precision RC kernels. No production library changes.
extern "C" {
__device__ __forceinline__ long offset(int b, int i, int j, int B, int n, int minor) {
    return minor ? ((long)i*n+j)*B+b : ((long)b*n+i)*n+j;
}
__device__ __forceinline__ double re(const double* y, long x, long N, int soa) {
    return y[soa ? x : 2*x];
}
__device__ __forceinline__ double im(const double* y, long x, long N, int soa) {
    return y[soa ? N+x : 2*x+1];
}
__device__ __forceinline__ void put(double* y, long x, long N, int soa, double r, double v) {
    y[soa ? x : 2*x]=r; y[soa ? N+x : 2*x+1]=v;
}
__global__ void begin_step(double* clock) {
    double t=clock[0], nominal=clock[3];
    double edge=(floor((t+1e-14)/2.5e-7)+1)*2.5e-7;
    clock[1]=fmax(0.0,fmin(nominal,fmin(edge-t,clock[2]-t)));
}
__global__ void advance(double* clock) { clock[0]+=clock[1]; }
__global__ void coefficients(const double* clock, const double* p, double fraction,
                             double* cr, double* ci) {
    double t=clock[0]+fraction*clock[1];
    int quarter=((int)floor((clock[0]+0.5*clock[1])/2.5e-7+1e-8))%4;
    int a=quarter<2, b=quarter==0 || quarter==3;
    double z=(t-p[0])/p[1], envelope=exp(-0.5*z*z);
    cr[0]=(!a)*p[2]*envelope; cr[1]=a*p[2]*envelope;
    cr[2]=b*p[3]*envelope; cr[3]=(!b)*p[3]*envelope;
    double phase=p[5]*sin(p[6]*t);
    cr[4]=p[4]*cos(phase); ci[4]=p[4]*sin(phase);
    ci[0]=ci[1]=ci[2]=ci[3]=0.0;
}
__global__ void build_h(const int* positions, const double* br, const double* bi,
                       const double* diagonal, const double* cr, const double* ci,
                       double* hz, double* hr, double* hi, int count, int n, int dense) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=count) return;
    int ij=positions[x], i=ij/n, j=ij%n;
    double r=(dense && i==j) ? diagonal[i] : 0., v=0.;
    for(int c=0;c<5;c++) {
        double a=cr[c], b=ci[c]*(i<=j ? 1. : -1.);
        double u=br[c*n*n+ij], w=bi[c*n*n+ij];
        r+=a*u-b*w; v+=a*w+b*u;
    }
    hz[2*x]=hr[x]=r; hz[2*x+1]=hi[x]=v;
}
}
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

extern "C" {
__global__ void stage_ptr(STATE* dest,const STATE* base,const STATE* k0,const STATE* k1,const STATE* k2,const STATE* k3,const STATE* k4,const STATE* k5,const STATE* k6,const double* w,const double* clock,long D,int row,int terms) {long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;const STATE* ks[7]={k0,k1,k2,k3,k4,k5,k6};double s=0.;for(int k=0;k<terms;k++)if(w[row*7+k]!=0.)s+=w[row*7+k]*(double)ks[k][x];dest[x]=(double)base[x]+clock[1]*s;}
__global__ void stage_c1(STATE* dest,const STATE* base,const STATE* k0,const STATE* k1,const STATE* k2,const STATE* k3,const STATE* k4,const STATE* k5,const STATE* k6,const double* clock,long D) {long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;double s=0.;s+=(0.161)*(double)k0[x];dest[x]=(double)base[x]+clock[1]*s;}
__global__ void stage_c2(STATE* dest,const STATE* base,const STATE* k0,const STATE* k1,const STATE* k2,const STATE* k3,const STATE* k4,const STATE* k5,const STATE* k6,const double* clock,long D) {long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;double s=0.;s+=(-0.0084806554923569887)*(double)k0[x];s+=(0.335480655492357)*(double)k1[x];dest[x]=(double)base[x]+clock[1]*s;}
__global__ void stage_c3(STATE* dest,const STATE* base,const STATE* k0,const STATE* k1,const STATE* k2,const STATE* k3,const STATE* k4,const STATE* k5,const STATE* k6,const double* clock,long D) {long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;double s=0.;s+=(2.8971530571054935)*(double)k0[x];s+=(-6.3594484899750752)*(double)k1[x];s+=(4.3622954328695815)*(double)k2[x];dest[x]=(double)base[x]+clock[1]*s;}
__global__ void stage_c4(STATE* dest,const STATE* base,const STATE* k0,const STATE* k1,const STATE* k2,const STATE* k3,const STATE* k4,const STATE* k5,const STATE* k6,const double* clock,long D) {long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;double s=0.;s+=(5.3258648284392569)*(double)k0[x];s+=(-11.748883564062828)*(double)k1[x];s+=(7.4955393428898365)*(double)k2[x];s+=(-0.092495066361755252)*(double)k3[x];dest[x]=(double)base[x]+clock[1]*s;}
__global__ void stage_c5(STATE* dest,const STATE* base,const STATE* k0,const STATE* k1,const STATE* k2,const STATE* k3,const STATE* k4,const STATE* k5,const STATE* k6,const double* clock,long D) {long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;double s=0.;s+=(5.8614554429464203)*(double)k0[x];s+=(-12.92096931784711)*(double)k1[x];s+=(8.1593678985761589)*(double)k2[x];s+=(-0.071584973281400996)*(double)k3[x];s+=(-0.028269050394068383)*(double)k4[x];dest[x]=(double)base[x]+clock[1]*s;}
__global__ void stage_c6(STATE* dest,const STATE* base,const STATE* k0,const STATE* k1,const STATE* k2,const STATE* k3,const STATE* k4,const STATE* k5,const STATE* k6,const double* clock,long D) {long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D)return;double s=0.;s+=(0.09646076681806523)*(double)k0[x];s+=(0.01)*(double)k1[x];s+=(0.4798896504144996)*(double)k2[x];s+=(1.3790085741037419)*(double)k3[x];s+=(-3.2900695154360808)*(double)k4[x];s+=(2.3247105240997739)*(double)k5[x];dest[x]=(double)base[x]+clock[1]*s;}
}

extern "C" {
__global__ void set_end(double* clock,double target){clock[2]=target;}
__global__ void sample_pop(const double* y,double* pops,double* times,const double* clock,int B,int n,int sample,int packed){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(x<B*n){int b=x/n,i=x%n;long ij=packed?i*n-i*(i-1)/2:i*n+i;
 pops[(long)sample*B*n+x]=y[ij*B+b];}
 if(x==0)times[sample]=clock[0];
}
}
// Upper triangle includes diagonal imaginary padding, always held at zero.
extern "C" {
__device__ __forceinline__ long pk(int i,int j,int n){if(i>j){int z=i;i=j;j=z;}return (long)i*n-i*(i+1)/2+j;}
__device__ __forceinline__ void readpk(const double* y,int i,int j,int b,int B,int n,long N,double &r,double &v){
 long x=pk(i,j,n)*B+b;r=y[x];v=(i>j?-1.:1.)*y[N+x];
}
__global__ void packed_product(const double* y,double* ar,double* ai,const int* ptr,const int* col,
 const double* hr,const double* hi,int B,int n,const double* clock,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,N=(long)B*n*n,P=(long)B*n*(n+1)/2;
 if(x>=N || (skip && clock[4]!=0.))return;
 int b=x%B,ij=x/B,i=ij/n,j=ij%n;double r=0.,v=0.;
 for(int p=ptr[i];p<ptr[i+1];p++){double u,w;readpk(y,col[p],j,b,B,n,P,u,w);r+=hr[p]*u-hi[p]*w;v+=hr[p]*w+hi[p]*u;}
 ar[x]=r;ai[x]=v;
}
__global__ void packed_finish(const double* y,double* out,const double* ar,const double* ai,
 const double* diag,const double* mask,const double* det,const double* rates,const int* tp,const int* tc,const double* tr,
 const int* rows,const int* cols,int B,int n,const double* clock,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,P=(long)B*n*(n+1)/2;
 if(x>=P || (skip && clock[4]!=0.))return;
 int b=x%B,p=x/B,i=rows[p],j=cols[p];
 long a=((long)i*n+j)*B+b,at=((long)j*n+i)*B+b;
 double u=y[x],v=y[P+x],d=diag[i]-diag[j]+det[b]*(mask[i]-mask[j]),loss=-.5*(rates[i]+rates[j]);
 double r=ai[a]+ai[at]+d*v+loss*u,w=ar[at]-ar[a]-d*u+loss*v;
 if(i==j){for(int k=tp[i];k<tp[i+1];k++)r+=tr[k]*y[pk(tc[k],tc[k],n)*B+b];w=0.;}
 out[x]=r;out[P+x]=w;
}
__global__ void packed_direct(const double* y,double* out,const int* ptr,const int* col,const double* hr,const double* hi,
 const double* diag,const double* mask,const double* det,const double* rates,const int* tp,const int* tc,const double* tr,
 const int* rows,const int* cols,int B,int n,const double* clock,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x,P=(long)B*n*(n+1)/2;
 if(x>=P || (skip && clock[4]!=0.))return;
 int b=x%B,p=x/B,i=rows[p],j=cols[p];double ar=0.,ai=0.,jr=0.,ji=0.;
 for(int k=ptr[i];k<ptr[i+1];k++){double u,v;readpk(y,col[k],j,b,B,n,P,u,v);ar+=hr[k]*u-hi[k]*v;ai+=hr[k]*v+hi[k]*u;}
 for(int k=ptr[j];k<ptr[j+1];k++){double u,v;readpk(y,col[k],i,b,B,n,P,u,v);jr+=hr[k]*u-hi[k]*v;ji+=hr[k]*v+hi[k]*u;}
 double u=y[x],v=y[P+x],d=diag[i]-diag[j]+det[b]*(mask[i]-mask[j]),loss=-.5*(rates[i]+rates[j]);
 double r=ai+ji+d*v+loss*u,w=jr-ar-d*u+loss*v;
 if(i==j){for(int k=tp[i];k<tp[i+1];k++)r+=tr[k]*y[pk(tc[k],tc[k],n)*B+b];w=0.;}
 out[x]=r;out[P+x]=w;
}
}

extern "C" {
__global__ void ls_update(double* y,double* r,const double* f,const double* clock,long D,double a,double b){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=D||clock[1]<=0.)return;
 double v=(a==0.?0.:a*r[x])+clock[1]*f[x];r[x]=v;y[x]+=b*v;
}
__global__ void table_h(const double* clock,const double* table,int steps,int stage,
 const int* positions,const double* br,const double* bi,double* hr,double* hi,int count,int n){
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=count)return;
 int step=min(steps-1,(int)clock[6]);long base=((long)step*7+stage)*10;
 int ij=positions[x],i=ij/n,j=ij%n;double r=0.,v=0.;
 for(int c=0;c<5;c++){double a=table[base+c],b=table[base+5+c]*(i<=j?1.:-1.);
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
