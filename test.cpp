#include<iostream>
#include<fstream>
#include<cstdlib>
#include<cmath>
#include<omp.h>
#include<cstring>

using namespace std;

void triSolve(double a[],double b[],double c[],double r[],double u[],int n)
{
	double beta,*gam;

	gam=(double *)malloc(n*sizeof(double));
	beta=b[0];
	u[0]=r[0]/beta;
	for (int j=1;j<n;j++)
	{
		gam[j]=c[j-1]/beta;
		beta=b[j]-a[j]*gam[j];
		u[j]=(r[j]-a[j]*u[j-1])/beta;
	}

	for (int j=n-2;j>=0;j--)
		u[j]-=gam[j+1]*u[j+1];

	free(gam);
}

void output(double *u,double *c,double *x,double *y,const char *fname,int hires)
{
	ofstream out(fname);

	//	out << N << endl;
	//	out << 0 << endl;

	int step=N/1000;

	if (hires)
		step=1;

	for (int i=0;i<=N;i+=step)
		for (int j=0;j<=N;j+=step)
		{
			int index=i*(N+1)+j;
			out << x[i] << "\t" << y[j] << "\t" << u[index] << "\t" << c[index] << endl;
		}

	out.close();

	cout << "Output : " << fname << endl;
}

double av(double *u)
{
	double sum=0;
	int count=0;
	int size=pow(N+1,2);

#pragma omp parallel for reduction(+:sum,count)
	for (int i=0;i<size;i++)
	{
		count++;
		sum+=u[i];
	}

	return sum/count;
}

double max(double *u)
{
	double maxVal=u[0];
	int size=pow(N+1,2);

#pragma omp parallel for reduction(max:maxVal)	
	for (int i=0;i<size;i++)
		maxVal=max(maxVal,u[i]);
	return maxVal;
}

double min(double *u)
{
	double minVal=u[0];
	int size=pow(N+1,2);

#pragma omp parallel for reduction(min:minVal)
	for (int i=0;i<size;i++)
		minVal=min(minVal,u[i]);
	return minVal;
}

void normalise(double *u,double umin,double umax)
{
	int size=pow(N+1,2);

#pragma omp parallel for
	for (int i=0;i<size;i++)
		u[i]=min(umax,max(umin,u[i]));
}

void diffusion(double *u,double *uHalf,double dt,double dx,double D)
{
	//	implicit in x, explicit in y
#pragma omp parallel for
	for (int j=0;j<=N;j++)
	{
		double a[N+1],b[N+1],c[N+1],r[N+1],sol[N+1];

		for (int i=0;i<=N;i++)
		{
			int index=i*(N+1)+j;
			int indexU=index+1; // up
			int indexD=index-1; // down
			int indexL=index-(N+1); // left
			int indexR=index+(N+1); // right
			double Uyy;

			if (i==0) // lhs
			{
				a[i]=0.0;
				b[i]=+D*2.0/pow(dx,2)+2.0/dt;
				c[i]=-D*2.0/pow(dx,2);
			}
			else if (i==N) // rhs
			{
				a[i]=-D*2.0/pow(dx,2);
				b[i]=+D*2.0/pow(dx,2)+2.0/dt;
				c[i]=0;
			}
			else
			{
				a[i]=-D*1.0/pow(dx,2);
				b[i]=+D*2.0/pow(dx,2)+2.0/dt;
				c[i]=-D*1.0/pow(dx,2);
			}

			if (j==0) // bottom
			{
				Uyy=2.0*(u[indexU]-u[index])/pow(dx,2);
			}
			else if (j==N) // top
			{
				Uyy=2.0*(u[indexD]-u[index])/pow(dx,2);
			}
			else
			{
				Uyy=(u[indexU]-2.0*u[index]+u[indexD])/pow(dx,2);
			}

			r[i]=2.0/dt*u[index]+D*Uyy;
		}
		triSolve(a,b,c,r,sol,N+1);

		for (int i=0;i<=N;i++)
		{
			int index=i*(N+1)+j;
			uHalf[index]=sol[i];
		}

	}

	//	explicit in x, implicit in y
#pragma omp parallel for
	for (int i=0;i<=N;i++)
	{
		double a[N+1],b[N+1],c[N+1],r[N+1],sol[N+1];

		for (int j=0;j<=N;j++)
		{
			int index=i*(N+1)+j;
			int indexU=index+1; // up
			int indexD=index-1; // down
			int indexL=index-(N+1); // left
			int indexR=index+(N+1); // right
			double Uxx;

			if (i==0) // lhs
			{
				Uxx=2.0*(uHalf[indexR]-uHalf[index])/pow(dx,2);
			}
			else if (i==N) // rhs
			{
				Uxx=2.0*(uHalf[indexL]-uHalf[index])/pow(dx,2);
			}
			else
			{
				Uxx=(uHalf[indexR]-2.0*uHalf[index]+uHalf[indexL])/pow(dx,2);
			}

			if (j==0) // bottom
			{
				a[j]=0.0;
				b[j]=+D*2.0/pow(dx,2)+2.0/dt;
				c[j]=-D*2.0/pow(dx,2);
			}
			else if (j==N) // top
			{
				a[j]=-D*2.0/pow(dx,2);
				b[j]=+D*2.0/pow(dx,2)+2.0/dt;
				c[j]=0;
			}
			else
			{
				a[j]=-D*1.0/pow(dx,2);
				b[j]=+D*2.0/pow(dx,2)+2.0/dt;
				c[j]=-D*1.0/pow(dx,2);
			}
			r[j]=2.0/dt*uHalf[index]+D*Uxx;
		}

		triSolve(a,b,c,r,sol,N+1);
		for (int j=0;j<=N;j++)
		{
			int index=i*(N+1)+j;
			u[index]=sol[j];
		}
	}
}

void advection(double *u,double *uH,double *x,double *y,double dt,double dx,
		double eta,double xi,double beam,double xs,double ys)
{
#pragma omp parallel for
	for (int j=0;j<=N;j++)
	{
		double a[N+1],b[N+1],c[N+1],r[N+1],sol[N+1];

		for (int i=0;i<=N;i++)
		{
			int index=i*(N+1)+j;
			int indexU=index+1; // up
			int indexD=index-1; // down
			int indexL=index-(N+1); // left
			int indexR=index+(N+1); // right

			double X=x[i]-xs,Y=y[j]-ys;
			double dist=sqrt(pow(beam,2)+pow(X,2)+pow(Y,2));
			double vx=(-eta*X-eta*xi*Y)/pow(dist,2);
			double vy=(eta*xi*X-eta*Y)/pow(dist,2);

			double Uy;

			if (i==0 || i==N)
			{
				a[i]=0;
				b[i]=2.0/dt;
				c[i]=0;
			}
			else
			{
				a[i]=-vx/2.0/dx;
				b[i]=2.0/dt;
				c[i]=+vx/2.0/dx;
			}

			if (j==0 || j==N)
				Uy=0;
			else
				Uy=(u[indexU]-u[indexD])/2.0/dx;

			r[i]=2.0/dt*u[index]-vy*Uy;
		}
		triSolve(a,b,c,r,sol,N+1);
		for (int i=0;i<=N;i++)
		{
			int index=i*(N+1)+j;
			uH[index]=sol[i];
		}
	}

#pragma omp parallel for
	for (int i=0;i<=N;i++)
	{
		double a[N+1],b[N+1],c[N+1],r[N+1],sol[N+1];

		for (int j=0;j<=N;j++)
		{
			int index=i*(N+1)+j;
			int indexU=index+1; // up
			int indexD=index-1; // down
			int indexL=index-(N+1); // left
			int indexR=index+(N+1); // right

			double X=x[i]-xs,Y=y[j]-ys;
			double dist=sqrt(pow(beam,2)+pow(X,2)+pow(Y,2));
			double vx=(-eta*X-eta*xi*Y)/pow(dist,2);
			double vy=(eta*xi*X-eta*Y)/pow(dist,2);
			double Ux;

			if (i==0 || i==N)
				Ux=0;
			else
				Ux=(uH[indexR]-uH[indexL])/2.0/dx;

			if (j==0 || j==N)
			{
				a[j]=0;
				b[j]=2.0/dt;
				c[j]=0;
			}
			else
			{
				a[j]=-vy/2.0/dx;
				b[j]=2.0/dt;
				c[j]=+vy/2.0/dx;
			}
			r[j]=2.0/dt*uH[index]-vx*Ux;
		}
		triSolve(a,b,c,r,sol,N+1);
		for (int j=0;j<=N;j++)
		{
			int index=i*(N+1)+j;
			u[index]=sol[j];
		}
	}
}

double fu(double u,double c,double q,double r,double f,double theta,double Da,double ua)
{
	double arr1,arr2;

	if (u<=ua)
	{
		arr1=0.0;arr2=0.0;
	}
	else
	{
		arr1=exp(-f/u);arr2=exp(-1.0/u);
	}

	return -q*r*Da*c*arr1/theta+Da*c*arr2/theta;
}

double fv(double u,double c,double q,double r,double f,double theta,double Da,double ua)
{
	double arr1,arr2;

	if (u<=ua)
	{
		arr1=0.0;arr2=0.0;
	}
	else
	{
		arr1=exp(-f/u);arr2=exp(-1.0/u);
	}

	return -r*Da*c*arr1-Da*c*arr2;
}

void rk(double *u0,double *c0,double *u1,double *c1,double dt,
		double q,double r,double f,double theta,double Da,double ua)
{
	int size=pow(N+1,2);

#pragma omp parallel for
	for (int i=0;i<size;i++)
	{
		double k1t, k2t, k3t, k4t, k1c, k2c, k3c, k4c;

		k1t = dt*fu(u0[i], c0[i],q,r,f,theta,Da,ua);
		k1c = dt*fv(u0[i], c0[i],q,r,f,theta,Da,ua);
		k2t = dt*fu(u0[i] + 0.5*k1t, c0[i] + 0.5*k1c,q,r,f,theta,Da,ua);
		k2c = dt*fv(u0[i] + 0.5*k1t, c0[i] + 0.5*k1c,q,r,f,theta,Da,ua);
		k3t = dt*fu(u0[i] + 0.5*k2t, c0[i] + 0.5*k2c,q,r,f,theta,Da,ua);
		k3c = dt*fv(u0[i] + 0.5*k2t, c0[i] + 0.5*k2c,q,r,f,theta,Da,ua);
		k4t = dt*fu(u0[i] + k3t, c0[i] + k3c,q,r,f,theta,Da,ua);
		k4c = dt*fv(u0[i] + k3t, c0[i] + k3c,q,r,f,theta,Da,ua);

		u1[i] = u0[i] + (k1t + 2.0*k2t + 2.0*k3t + k4t) / 6.0;
		c1[i] = c0[i] + (k1c + 2.0*k2c + 2.0*k3c + k4c) / 6.0;
	}
}   

double findErr(double *uF,double *uH2,double *cF,double *cH2)
{
	double sum=0;
#pragma omp parallel for reduction(+:sum)
	for (int i=0;i<(N+1)*(N+1);i++)
		sum+=abs(uF[i]-uH2[i])+abs(cF[i]-cH2[i]);
	return sum/pow(N+1,2)/2.0;
}

double findErrL2(double *uF,double *uH2,double *cF,double *cH2)
{
	double sum1=0,sum2=0;

#pragma omp parallel for reduction(+:sum1,sum2)
	for (int i=0;i<(N+1)*(N+1);i++)
	{
		sum1+=pow(uF[i]-uH2[i],2);
		sum2+=pow(cF[i]-cH2[i],2);
	}
	return max(sqrt(sum1),sqrt(sum2));
}

int main(int argc,char ** argv)
{
	double L=10;
	double dx=1.0*L/N,dy=1.0*L/N;
	double eta=0.1,xi=20,Pe=2000,Le=1;
	double t=0,dt0=0.0001,dt=dt0;
	double q=1,r=1,f=2,Da=10,theta=1;
	int size=pow(N+1,2);
	double ua=0,A=1,sigma=0.5,beam=0.1;
	double pi=4.0*atan(1.0);
	double invPe=1.0/Pe;
	int hires=0;

	double u[size],c[size];
	double uWKS[size],cWKS[size];
	double uF[size],cF[size];
	double uF_WKS[size],cF_WKS[size];
	double uH1[size],cH1[size];
	double uH1_WKS[size],cH1_WKS[size];
	double uH2[size],cH2[size];
	double uH2_WKS[size],cH2_WKS[size];
	double x[N+1],y[N+1];
	double u0[size],u1[size],u2[size];
	double c0[size],c1[size],c2[size];

	double tmax=200;

	int scenario=2;

	cout << "here" << endl;

	int nmins=10000;

	char oname[1000],pname[1000];

	for (int i=1;i<argc;i+=2)
	{
		if (strcmp(argv[i],"nmins")==0)
			nmins=atoi(argv[i+1]);
		else if (strcmp(argv[i],"Pe")==0)
		{
			Pe=atof(argv[i+1]);
			invPe=1.0/Pe;
		}
		else if (strcmp(argv[i],"eta")==0)
			eta=atof(argv[i+1]);
		else if (strcmp(argv[i],"tmax")==0)
			tmax=atof(argv[i+1]);
		else if (strcmp(argv[i],"A")==0)
			A=atof(argv[i+1]);
		else if (strcmp(argv[i],"sigma")==0)
			sigma=atof(argv[i+1]);
		else if (strcmp(argv[i],"L")==0)
		{
			L=atof(argv[i+1]);dx=1.0*L/N;dy=1.0*L/N;
		}
		else if (strcmp(argv[i],"noD")==0)
		{
			invPe=0; // infinite Peclet number
		}
		else if (strcmp(argv[i],"Da")==0)
		{
			Da=atof(argv[i+1]);
		}
		else if (strcmp(argv[i],"hires")==0)
		{
			hires=1;
		}
		else if (strcmp(argv[i],"xi")==0)
		{
			xi=atof(argv[i+1]);
		}
		else if (strcmp(argv[i],"f")==0)
		{
			f=atof(argv[i+1]);
		}
		else if (strcmp(argv[i],"q")==0)
		{
			q=atof(argv[i+1]);
		}
		else if (strcmp(argv[i],"Theta")==0)
		{
			theta=atof(argv[i+1]);
		}
		else if (strcmp(argv[i],"r")==0)
		{
			r=atof(argv[i+1]);
		}
		else if (strcmp(argv[i],"scen")==0)
		{
			scenario=atoi(argv[i+1]);
		}
		else
		{
			cout << "Parameter " << argv[i] << " not recognised" << endl;
			exit(0);
		}
	}

	for (int i=0;i<=N;i++)
	{
		x[i]=i*dx-L/2;
		for (int j=0;j<=N;j++)
		{
			int index=i*(N+1)+j;
			y[j]=j*dx-L/2;
			u[index]=A*exp(-(pow(x[i],2)+pow(y[j],2))/pow(sigma,2))+ua;
			c[index]=1.0;
			u0[index]=u[index];c0[index]=c[index];
			/*
			   if (pow(x[i],2)+pow(y[j],2)<=pow(sigma,2))
			   u[index]=A+ua;
			   else
			   u[index]=ua;
			   */
		}
	}

	cout << av(u) << " " << av(c) << endl;

	double fac=0;
	sprintf(oname,"output_omp-N-%i",N);sprintf(pname,"profile_omp-N-%i",N);
	for (int i=1;i<argc;i+=2) 
	{
		sprintf(oname,"%s-%s-%s",oname,argv[i],argv[i+1]);
		sprintf(pname,"%s-%s-%s",pname,argv[i],argv[i+1]);
	}
	sprintf(oname,"%s.dat",oname);sprintf(pname,"%s.dat",pname);

	ofstream out(oname);

	double sbuff=0.9; // safety buffer
	double st=omp_get_wtime();
	int firstRun=1;

	cout << omp_get_wtime() << endl;
	int loop=0;
	while (t<tmax && omp_get_wtime()-st<sbuff*nmins*60)
	{
		double err,eps=1e-6;
		double xs,ys;
		/*
		   if (sin(2.0*pi*t)<0)
		   {
		   xs=+1;ys=0;
		   }
		   else
		   {
		   xs=-1;ys=0;
		   }
		   */
		/*
		   if (scenario==1)
		   {
		   double period=0.5;
		   if (int(t/period)%2==0)
		   {
		   xs=1;ys=0;
		   }
		   else
		   {
		   xs=-1;ys=0;
		   }
		   }
		   else if (scenario==2)
		   {
		   double period=0.5;
		   if (int(t/period)%2==0)
		   {
		   xs=0;ys=1;
		   }
		   else
		   {
		   xs=0;ys=-1;
		   }
		   }
		   else if (scenario==3)
		   {
		   double period=0.5;
		   if (int(t/period)%2==0)
		   {
		   xs=sqrt(0.5);ys=sqrt(0.5);
		   }
		   else
		   {
		   xs=-sqrt(0.5);ys=-sqrt(0.5);
		   }
		   }
		   */
		double period=1.0/scenario;
		int index=int(t/period)%scenario;
		xs=cos(2.0*index*pi/scenario);
		ys=sin(2.0*index*pi/scenario);

		// full step
		rk(u,c,uF,cF,dt,q,r,f,theta,Da,ua);
		diffusion(uF,uF_WKS,dt,dx,Le*invPe);
		diffusion(cF,cF_WKS,dt,dx,invPe);
		advection(uF,uF_WKS,x,y,dt,dx,eta,xi,beam,xs,ys);
		advection(cF,cF_WKS,x,y,dt,dx,eta,xi,beam,xs,ys);
		normalise(uF,ua,1e8);
		normalise(cF,0.0,1.0);
		// first half step
		rk(u,c,uH1,cH1,0.5*dt,q,r,f,theta,Da,ua);
		diffusion(uH1,uH1_WKS,0.5*dt,dx,Le*invPe);
		diffusion(cH1,cH1_WKS,0.5*dt,dx,invPe);
		advection(uH1,uH1_WKS,x,y,0.5*dt,dx,eta,xi,beam,xs,ys);
		advection(cH1,cH1_WKS,x,y,0.5*dt,dx,eta,xi,beam,xs,ys);
		normalise(uH1,ua,1e8);
		normalise(cH1,0.0,1.0);
		// second half step
		rk(uH1,cH1,uH2,cH2,0.5*dt,q,r,f,theta,Da,ua);
		diffusion(uH2,uH2_WKS,0.5*dt,dx,Le*invPe);
		diffusion(cH2,cH2_WKS,0.5*dt,dx,invPe);
		advection(uH2,uH2_WKS,x,y,0.5*dt,dx,eta,xi,beam,xs,ys);
		advection(cH2,cH2_WKS,x,y,0.5*dt,dx,eta,xi,beam,xs,ys);
		normalise(uH2,ua,1e8);
		normalise(cH2,0.0,1.0);
		err=findErr(uF,uH2,cF,cH2);
		cout << "t = " << t << " dt = " << dt << " err = " << err << endl;

		if (err<=eps)
		{
			t+=dt;
#pragma omp parallel for
			for (int i=0;i<(N+1)*(N+1);i++)
			{
				u[i]=uH2[i];c[i]=cH2[i];
			}
			cout << t << " " << dt << " " << av(u) << " " << av(c);
			cout << " max = " << max(u) << " min = " << min(u) <<  endl;
			if (err<0.25*eps)
				dt*=2;
			dt=min(dt,tmax-t);
			double err1=findErr(uF,uH2,cF,cH2);
			double err2=findErrL2(uF,uH2,cF,cH2);
			out << t << " " << av(u) << " " << av(c) << " " << err1 << " " << err2 << endl;

		}
		else
			dt/=2.0;

		if (t>=fac)
		{
			//char fname[300];
			//sprintf(fname,"%s-%04i",pname,int(10*fac+0.5));
			//output(u,c,x,y,fname,hires);
			//fac+=0.1;
			output(u,c,x,y,pname,hires);
			fac+=tmax/10;
		}
	}


	out.close();

	output(u,c,x,y,pname,hires);


}

