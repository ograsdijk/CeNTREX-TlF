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
// Direct output ownership: evaluate A_ij and A_ji without atomic accumulation.
__global__ void fused_rhs(const double* y, double* out,
                         const int* ptr, const int* col, const double* hr, const double* hi,
                         const double* diag, const double* mask, const double* det,
                         const double* rates, const int* tp, const int* tc, const double* tr,
                         int B, int n, int soa, int minor) {
    long N=(long)B*n*n, x=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=N) return;
    int b=minor ? x%B : x/(n*n), ij=minor ? x/B : x%(n*n), i=ij/n, j=ij%n;
    double ar=0., ai=0., jr=0., ji=0.;
    for(int p=ptr[i];p<ptr[i+1];p++) {
        long k=offset(b,col[p],j,B,n,minor);
        double u=re(y,k,N,soa), v=im(y,k,N,soa);
        ar+=hr[p]*u-hi[p]*v; ai+=hr[p]*v+hi[p]*u;
    }
    for(int p=ptr[j];p<ptr[j+1];p++) {
        long k=offset(b,col[p],i,B,n,minor);
        double u=re(y,k,N,soa), v=im(y,k,N,soa);
        jr+=hr[p]*u-hi[p]*v; ji+=hr[p]*v+hi[p]*u;
    }
    double u=re(y,x,N,soa), v=im(y,x,N,soa);
    double d=diag[i]-diag[j]+det[b]*(mask[i]-mask[j]);
    double loss=-0.5*(rates[i]+rates[j]);
    double r=ai+ji+d*v+loss*u, w=jr-ar-d*u+loss*v;
    if(i==j) {
        for(int p=tp[i];p<tp[i+1];p++) r+=tr[p]*re(y,offset(b,tc[p],tc[p],B,n,minor),N,soa);
        w=0.;
    }
    put(out,x,N,soa,r,w);
}
__global__ void sparse_product(const double* y, double* ar, double* ai,
                              const int* ptr, const int* col, const double* hr, const double* hi,
                              int B, int n, int soa, int minor) {
    long N=(long)B*n*n, x=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=N) return;
    int b=minor ? x%B : x/(n*n), ij=minor ? x/B : x%(n*n), i=ij/n, j=ij%n;
    double r=0., w=0.;
    for(int p=ptr[i];p<ptr[i+1];p++) {
        long k=offset(b,col[p],j,B,n,minor);
        double u=re(y,k,N,soa), v=im(y,k,N,soa);
        r+=hr[p]*u-hi[p]*v; w+=hr[p]*v+hi[p]*u;
    }
    ar[x]=r; ai[x]=w;
}
__global__ void finish_rhs(const double* y, double* out, const double* ar, const double* ai,
                          const double* az, int complex_a,
                          const double* diag, const double* mask, const double* det,
                          const double* rates, const int* tp, const int* tc, const double* tr,
                          int B, int n, int soa, int minor, int split_diag) {
    long N=(long)B*n*n, x=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=N) return;
    int b=minor ? x%B : x/(n*n), ij=minor ? x/B : x%(n*n), i=ij/n, j=ij%n;
    long xt=offset(b,j,i,B,n,minor);
    double r1=complex_a ? az[2*x] : ar[x], v1=complex_a ? az[2*x+1] : ai[x];
    double r2=complex_a ? az[2*xt] : ar[xt], v2=complex_a ? az[2*xt+1] : ai[xt];
    double u=re(y,x,N,soa), v=im(y,x,N,soa);
    double d=det[b]*(mask[i]-mask[j])+(split_diag ? diag[i]-diag[j] : 0.);
    double loss=-0.5*(rates[i]+rates[j]);
    double r=v1+v2+d*v+loss*u, w=r2-r1-d*u+loss*v;
    if(i==j) {
        for(int p=tp[i];p<tp[i+1];p++) r+=tr[p]*re(y,offset(b,tc[p],tc[p],B,n,minor),N,soa);
        w=0.;
    }
    put(out,x,N,soa,r,w);
}
// Arbitrary two-sided dense commutator, baseline (Hermitian input required for diag).
__global__ void finish_two(const double* y, double* out, const double* comm,
                          const double* mask, const double* det, const double* rates,
                          const int* tp, const int* tc, const double* tr, int B, int n) {
    long N=(long)B*n*n, x=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=N) return;
    int b=x/(n*n), ij=x%(n*n), i=ij/n,j=ij%n;
    double u=y[2*x], v=y[2*x+1], d=det[b]*(mask[i]-mask[j]);
    double loss=-.5*(rates[i]+rates[j]);
    double r=comm[2*x+1]+d*v+loss*u, w=-comm[2*x]-d*u+loss*v;
    if(i==j) {
        for(int p=tp[i];p<tp[i+1];p++) r+=tr[p]*y[2*offset(b,tc[p],tc[p],B,n,0)];
        w=0.;
    }
    out[2*x]=r; out[2*x+1]=w;
}
__global__ void combine(double* dest, const double* base, const double* k1, const double* k2,
                        const double* k3, const double* k4, const double* clock,
                        double a, double b, double c, double d, long count) {
    long x=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=count) return;
    double s=0.;
    if(a!=0.) s+=a*k1[x]; if(b!=0.) s+=b*k2[x];
    if(c!=0.) s+=c*k3[x]; if(d!=0.) s+=d*k4[x];
    dest[x]=base[x]+clock[1]*s;
}
}
