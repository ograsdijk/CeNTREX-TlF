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
