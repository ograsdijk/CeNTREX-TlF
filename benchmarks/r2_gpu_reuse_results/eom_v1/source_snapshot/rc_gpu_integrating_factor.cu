extern "C" {
__global__ void if_factors(const double* diag,const double* mask,const double* det,const double* rates,
 const double* clock,double fraction,double* qr,double* qi,double* invnorm,int B,int n){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=B*n)return;
 int b=x%B,i=x/B;double t=clock[1]*fraction;
 double omega=diag[i]-diag[0]+det[b]*mask[i],a=exp(-.5*rates[i]*t),s,c;
 sincos(-omega*t,&s,&c);qr[x]=a*c;qi[x]=a*s;invnorm[x]=exp(rates[i]*t);
}
__global__ void if_rotate(const double* input,double* out,const double* qr,const double* qi,const double* invnorm,
 int B,int n,int inverse){
 long N=(long)B*n*n,x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=N)return;
 int b=x%B,ij=x/B,i=ij/n,j=ij%n;long a=(long)i*B+b,c=(long)j*B+b;
 double r=qr[a]*qr[c]+qi[a]*qi[c],v=qi[a]*qr[c]-qr[a]*qi[c];
 if(inverse){double scale=invnorm[a]*invnorm[c];r*=scale;v*=-scale;}
 double u=input[x],w=input[N+x];out[x]=r*u-v*w;out[N+x]=r*w+v*u;
}
}
