#include<iostream>
#include<fstream>
#include<cstdlib>
#include<cmath>
#include<omp.h>
#include<cstring>
#include<vector>
#include<string>

using namespace std;

inline int idx(int i,int j,int n) {return i*(n+1)+j;}

void triSolve(vector<double> a,vector<double> b,vector<double> c,vector<double> r,vector<double> &u,int n)
{
	double beta;
	vector<double> gam(n);

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

}

void output(vector<double> u,vector<double> c,vector<double> x,vector<double> y,string fname,int hires,int N)
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
			out << x[i] << "\t" << y[j] << "\t" << u[idx(i,j,N)] << "\t" << c[idx(i,j,N)] << endl;
		}

	out.close();

	cout << "Output : " << fname << endl;
}

double av(vector<double> u)
{
	double sum=0;
	int count=0;
	int size=u.size();

#pragma omp parallel for reduction(+:sum,count)
	for (int i=0;i<size;i++)
	{
		count++;
		sum+=u[i];
	}

	return sum/count;
}

double max(vector<double> u)
{
	double maxVal=u[0];
	int size=u.size();

#pragma omp parallel for reduction(max:maxVal)	
	for (int i=0;i<size;i++)
		maxVal=max(maxVal,u[i]);
	return maxVal;
}

double min(vector<double> u)
{
	double minVal=u[0];
	int size=u.size();

#pragma omp parallel for reduction(min:minVal)
	for (int i=0;i<size;i++)
		minVal=min(minVal,u[i]);
	return minVal;
}

void normalise(vector<double> &u,double umin,double umax)
{
	int size=u.size();

#pragma omp parallel for
	for (int i=0;i<size;i++)
		u[i]=min(umax,max(umin,u[i]));
}

void diffusion(vector<double> &u,vector<double> uHalf,double dt,double dx,double D,int N)
{
	//	implicit in x, explicit in y
#pragma omp parallel for
	for (int j=0;j<=N;j++)
	{
		vector<double> a(N+1),b(N+1),c(N+1),r(N+1),sol(N+1);

		for (int i=0;i<=N;i++)
		{
			int index=idx(i,j,N);
			int indexU=idx(i,j+1,N); // up
			int indexD=idx(i,j-1,N); // down
						 //int indexL=idx(i-1,j,N); // left
						 //int indexR=idx(i+1,j,N); // right
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
			uHalf[idx(i,j,N)]=sol[i];
		}

	}

	//	explicit in x, implicit in y
#pragma omp parallel for
	for (int i=0;i<=N;i++)
	{
		vector<double> a(N+1),b(N+1),c(N+1),r(N+1),sol(N+1);

		for (int j=0;j<=N;j++)
		{
			int index=idx(i,j,N);
			//int indexU=idx(i,j+1,N); // up
			//int indexD=idx(i,j-1,N); // down
			int indexL=idx(i-1,j,N); // left
			int indexR=idx(i+1,j,N); // right
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
			u[idx(i,j,N)]=sol[j];
		}
	}
}

void advection(vector<double> &u,vector<double> uH,vector<double> x,vector<double> y,double dt,double dx,
		double eta,double xi,double width,int advType,double xs,double ys,int N)
{
#pragma omp parallel for
	for (int j=0;j<=N;j++)
	{
		vector<double> a(N+1),b(N+1),c(N+1),r(N+1),sol(N+1);

		for (int i=0;i<=N;i++)
		{
			int index=idx(i,j,N);
			int indexU=idx(i,j+1,N); // up
			int indexD=idx(i,j-1,N); // down
						 //int indexL=idx(i-1,j,N); // left
						 //int indexR=idx(i+1,j,N); // right

			double X=x[i]-xs,Y=y[j]-ys;
			double dist,vx,vy;

			if (advType==0)
			{
				dist=sqrt(pow(width,2)+pow(X,2)+pow(Y,2));
				vx=(-eta*X-eta*xi*Y)/pow(dist,2);
				vy=(eta*xi*X-eta*Y)/pow(dist,2);
			}
			else
			{
				dist=sqrt(pow(X,2)+pow(Y,2));
				if (dist==0)
				{
					vx=(-eta*X-eta*xi*Y)/pow(width,2);
					vy=(+eta*xi*X-eta*Y)/pow(width,2);
				}
				else
				{
					vx=(-eta*X-eta*xi*Y)*(1.0-exp(pow(dist/width,2)))/pow(dist,2);
					vy=(+eta*xi*X-eta*Y)*(1.0-exp(pow(dist/width,2)))/pow(dist,2);
				}
			}

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
			uH[idx(i,j,N)]=sol[i];
		}
	}

#pragma omp parallel for
	for (int i=0;i<=N;i++)
	{
		vector<double> a(N+1),b(N+1),c(N+1),r(N+1),sol(N+1);

		for (int j=0;j<=N;j++)
		{
			int index=idx(i,j,N);
			//int indexU=idx(i,j+1,N); // up
			//int indexD=idx(i,j-1,N); // down
			int indexL=idx(i-1,j,N); // left
			int indexR=idx(i+1,j,N); // right

			double X=x[i]-xs,Y=y[j]-ys;
			double dist,vx,vy;

			if (advType==0)
			{
				dist=sqrt(pow(width,2)+pow(X,2)+pow(Y,2));
				vx=(-eta*X-eta*xi*Y)/pow(dist,2);
				vy=(eta*xi*X-eta*Y)/pow(dist,2);
			}
			else
			{
				dist=sqrt(pow(X,2)+pow(Y,2));
				if (dist==0)
				{
					vx=(-eta*X-eta*xi*Y)/pow(width,2);
					vy=(+eta*xi*X-eta*Y)/pow(width,2);
				}
				else
				{
					vx=(-eta*X-eta*xi*Y)*(1.0-exp(pow(dist/width,2)))/pow(dist,2);
					vy=(+eta*xi*X-eta*Y)*(1.0-exp(pow(dist/width,2)))/pow(dist,2);
				}
			}

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
			u[idx(i,j,N)]=sol[j];
		}
	}
}

double fu(double u,double c,double q,double r,double f,double theta,double ua)
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

	return -q*r*c*arr1/theta+c*arr2/theta;
}

double fv(double u,double c,double q,double r,double f,double theta,double ua)
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

	return -r*c*arr1-c*arr2;
}

void rk(vector<double> u0,vector<double> c0,vector<double> &u1,vector<double> &c1,double dt,
		double q,double r,double f,double theta,double ua)
{
	int size=u0.size();

#pragma omp parallel for
	for (int i=0;i<size;i++)
	{
		double k1t, k2t, k3t, k4t, k1c, k2c, k3c, k4c;

		k1t = dt*fu(u0[i], c0[i],q,r,f,theta,ua);
		k1c = dt*fv(u0[i], c0[i],q,r,f,theta,ua);
		k2t = dt*fu(u0[i] + 0.5*k1t, c0[i] + 0.5*k1c,q,r,f,theta,ua);
		k2c = dt*fv(u0[i] + 0.5*k1t, c0[i] + 0.5*k1c,q,r,f,theta,ua);
		k3t = dt*fu(u0[i] + 0.5*k2t, c0[i] + 0.5*k2c,q,r,f,theta,ua);
		k3c = dt*fv(u0[i] + 0.5*k2t, c0[i] + 0.5*k2c,q,r,f,theta,ua);
		k4t = dt*fu(u0[i] + k3t, c0[i] + k3c,q,r,f,theta,ua);
		k4c = dt*fv(u0[i] + k3t, c0[i] + k3c,q,r,f,theta,ua);

		u1[i] = u0[i] + (k1t + 2.0*k2t + 2.0*k3t + k4t) / 6.0;
		c1[i] = c0[i] + (k1c + 2.0*k2c + 2.0*k3c + k4c) / 6.0;
	}
}   

double findErr(vector<double> uF,vector<double> uH2,vector<double> cF,vector<double> cH2)
{
	double sum=0;
	int size=uF.size();

#pragma omp parallel for reduction(+:sum)
	for (int i=0;i<size;i++)
		sum+=abs(uF[i]-uH2[i])+abs(cF[i]-cH2[i]);
	return sum/size/2.0;
}

double findErrNew(vector<double> &uF,vector<double> &uH2,vector<double> &cF,vector<double> &cH2,
		double eps,double atol,double rtol)
{
	double sum=0;
	int size=uF.size();

#pragma omp parallel for reduction(+:sum)
	for (int i=0;i<size;i++)
	{
		sum+=abs(uF[i]-uH2[i])/(atol+rtol*max(abs(uF[i]),abs(uH2[i])));
		sum+=abs(cF[i]-cH2[i])/(atol+rtol*max(abs(cF[i]),abs(cH2[i])));
	}
	return sum*eps/2.0/size;
}

double findErrL2(vector<double> uF,vector<double> uH2,vector<double> cF,vector<double> cH2)
{
	double sum1=0,sum2=0;
	int size=uF.size();

#pragma omp parallel for reduction(+:sum1,sum2)
	for (int i=0;i<size;i++)
	{
		sum1+=pow(uF[i]-uH2[i],2);
		sum2+=pow(cF[i]-cH2[i],2);
	}
	return max(sqrt(sum1),sqrt(sum2));
}

int main(int argc,char ** argv)
{
	int N=1000;
	double L=10;
	double dx=1.0*L/N;
	double eta=0.1,xi=20,Pe=2000,Le=1;
	double t=0,dt0=0.0001,dt=dt0;
	double q=1,r=1,f=2,theta=1;
	int size=pow(N+1,2);
	double ua=0,A=1,sigma=0.5,width=0.1;
	double pi=4.0*atan(1.0);
	double invPe=1.0/Pe;
	int hires=0;
	double T0=1;
	int scenario=2;
	double atol=1e-8,rtol=1e-5;
	int total_steps=0,total_rejected=0;
	int advType=0;
	
	vector<double> u(size),c(size);
	vector<double> uWKS(size),cWKS(size);
	vector<double> uF(size),cF(size);
	vector<double> uF_WKS(size),cF_WKS(size);
	vector<double> uH1(size),cH1(size);
	vector<double> uH1_WKS(size),cH1_WKS(size);
	vector<double> uH2(size),cH2(size);
	vector<double> uH2_WKS(size),cH2_WKS(size);
	vector<double> x(N+1),y(N+1);

	double tmax=200;

	double st=omp_get_wtime();

	cout << "here" << endl;

	int nmins=10000;

	string oname,pname;

	for (int i=1;i<argc;i+=2)
	{
		string arg=argv[i];
		if (arg=="nmins")
			nmins=atoi(argv[i+1]);
		else if (arg=="width")
			width=atof(argv[i+1]);
		else if (arg=="advType")
			advType=atoi(argv[i+1]);
		else if (arg=="Pe")
		{
			Pe=atof(argv[i+1]);
			invPe=1.0/Pe;
		}
		else if (arg=="eta")
			eta=atof(argv[i+1]);
		else if (arg=="tmax")
			tmax=atof(argv[i+1]);
		else if (arg=="A")
			A=atof(argv[i+1]);
		else if (arg=="sigma")
			sigma=atof(argv[i+1]);
		else if (arg=="L")
		{
			L=atof(argv[i+1]);dx=1.0*L/N;
		}
		else if (arg=="noD")
		{
			invPe=0; // infinite Peclet number
		}
		else if (arg=="hires")
		{
			hires=1;
		}
		else if (arg=="xi")
		{
			xi=atof(argv[i+1]);
		}
		else if (arg=="f")
		{
			f=atof(argv[i+1]);
		}
		else if (arg=="q")
		{
			q=atof(argv[i+1]);
		}
		else if (arg=="Theta")
		{
			theta=atof(argv[i+1]);
		}
		else if (arg=="r")
		{
			r=atof(argv[i+1]);
		}
		else if (arg=="scen")
		{
			scenario=atoi(argv[i+1]);
		}
		else if (arg=="atol")
		{
			atol=atof(argv[i+1]);
		}
		else if (arg=="rtol")
		{
			rtol=atof(argv[i+1]);
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
			int index=idx(i,j,N);
			y[j]=j*dx-L/2;
			u[index]=A*exp(-(pow(x[i],2)+pow(y[j],2))/pow(sigma,2))+ua;
			c[index]=1.0;
		}
	}

	cout << av(u) << " " << av(c) << endl;

	double fac=0;
	oname="output_new-N-"+to_string(N);pname="profile_new-N-"+to_string(N);
	for (int i=1;i<argc;i+=2) 
	{
		oname+="-"+string(argv[i])+"-"+string(argv[i+1]);
		pname+="-"+string(argv[i])+"-"+string(argv[i+1]);
	}
	oname+=".dat";pname+=".dat";

	ofstream out(oname);

	double sbuff=0.9; // safety buffer
	double period=1.0*T0/scenario,nextBlink=period;
	double xs,ys;

	cout << omp_get_wtime() << endl;
	while (t<tmax && omp_get_wtime()-st<sbuff*nmins*60)
	{
		double err,eps=1e-6;
		int index=int(t/period)%scenario;
		xs=cos(2.0*index*pi/scenario);
		ys=sin(2.0*index*pi/scenario);

		dt=max(dt0,dt);
		double tend=min(nextBlink,tmax);
		while (t<tend)
		{
			dt=min(dt,tend-t);

			// full step
			rk(u,c,uF,cF,dt,q,r,f,theta,ua);
			diffusion(uF,uF_WKS,dt,dx,Le*invPe,N);
			diffusion(cF,cF_WKS,dt,dx,invPe,N);
			advection(uF,uF_WKS,x,y,dt,dx,eta,xi,width,advType,xs,ys,N);
			advection(cF,cF_WKS,x,y,dt,dx,eta,xi,width,advType,xs,ys,N);
			normalise(uF,ua,1e8);
			normalise(cF,0.0,1.0);
			// first half step
			rk(u,c,uH1,cH1,0.5*dt,q,r,f,theta,ua);
			diffusion(uH1,uH1_WKS,0.5*dt,dx,Le*invPe,N);
			diffusion(cH1,cH1_WKS,0.5*dt,dx,invPe,N);
			advection(uH1,uH1_WKS,x,y,0.5*dt,dx,eta,xi,width,advType,xs,ys,N);
			advection(cH1,cH1_WKS,x,y,0.5*dt,dx,eta,xi,width,advType,xs,ys,N);
			normalise(uH1,ua,1e8);
			normalise(cH1,0.0,1.0);
			// second half step
			rk(uH1,cH1,uH2,cH2,0.5*dt,q,r,f,theta,ua);
			diffusion(uH2,uH2_WKS,0.5*dt,dx,Le*invPe,N);
			diffusion(cH2,cH2_WKS,0.5*dt,dx,invPe,N);
			advection(uH2,uH2_WKS,x,y,0.5*dt,dx,eta,xi,width,advType,xs,ys,N);
			advection(cH2,cH2_WKS,x,y,0.5*dt,dx,eta,xi,width,advType,xs,ys,N);
			normalise(uH2,ua,1e8);
			normalise(cH2,0.0,1.0);
			//err=findErr(uF,uH2,cF,cH2);
			err=findErrNew(uF,uH2,cF,cH2,eps,atol,rtol);
			cout << "t = " << t << " dt = " << dt << " err = " << err << endl;

			if (err<=eps)
			{
				total_steps++;
				t+=dt;
#pragma omp parallel for
				for (int i=0;i<size;i++)
				{
					u[i]=uH2[i];c[i]=cH2[i];
				}
				cout << t << " " << dt << " " << av(u) << " " << av(c);
				cout << " max = " << max(u) << " min = " << min(u) <<  endl;

				if (err<0.25*eps)
					dt*=2;
				double err1=findErr(uF,uH2,cF,cH2);
				double err2=findErrL2(uF,uH2,cF,cH2);
				out << t << " " << av(u) << " " << av(c) << " " << err1 << " " << err2 << endl;

			}
			else
			{
				dt/=2.0;
				total_rejected++;
			}

			if (t>=fac)
			{
				//output(u,c,x,y,pname,hires,N);
				fac+=tmax/10;
			}
		}
		t=tend; // avoid round off errors
		nextBlink+=period;
		cout << "next blink at t = " << nextBlink << endl;

	}


	cout << "wall clock = " << int(omp_get_wtime()-st+0.5) << endl;
	printf("average timestep = %e\n",1.0*tmax/total_steps);
	cout << "number of rejected steps = " << total_rejected << endl;

	out.close();

	output(u,c,x,y,pname,hires,N);


}

