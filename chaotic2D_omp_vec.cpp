#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<string>

using namespace std;

inline int idx(int i,int j,int N) {return i*(N+1)+j;}

void output(vector<double> u,vector<double> c,vector<double> x,vector<double> y,int N,string fname)
{
	ofstream out(fname);
	int step=N/1000;

	for (int i=0;i<=N;i+=step)
		for (int j=0;j<=N;j+=step)
		{
			out << x[i] << " " << y[j] << " " << u[idx(i,j,N)] << " " << c[idx(i,j,N)] << endl;
		}

	out.close();

	cout << "Output : " << fname << endl;
}

double av(vector<double> u)
{
	double sum=0;
	for (int i=0;i<u.size();i++)
		sum+=u[i];
	return sum/u.size();
}

double clamp(double u,double umin,double umax)
{
	return min(max(u,umin),umax);
}

double fu(double u,double c,double q,double r,double f,double theta,double ua)
{
	double arr1,arr2;

	if (u<ua)
	{
		arr1=0.0;arr2=0.0;
	}
	else
	{
		arr1=exp(-f/u);arr2=exp(-1.0/u);
	}

//	return -q*r*c*arr1+c*arr2;
	return -u*u;
}

double fc(double u,double c,double q,double r,double f,double theta,double ua)
{
        double arr1,arr2;

        if (u<ua)
        {
                arr1=0.0;arr2=0.0;
        }
        else
        {
                arr1=exp(-f/u);arr2=exp(-1.0/u);
        }

        //return -theta*r*c*arr1-theta*c*arr2;
	return c*c;
}

void rk(vector<double> u0,vector<double> c0,vector<double> &u1,vector<double> &c1,double dt,
		double q,double r,double f,double theta,double ua)
{
	for (int i=0;i<u0.size();i++)
	{
		 double k1t, k2t, k3t, k4t, k1c, k2c, k3c, k4c;

                k1t = dt*fu(u0[i], c0[i],q,r,f,theta,ua);
                k1c = dt*fc(u0[i], c0[i],q,r,f,theta,ua);
                k2t = dt*fu(u0[i] + 0.5*k1t, c0[i] + 0.5*k1c,q,r,f,theta,ua);
                k2c = dt*fc(u0[i] + 0.5*k1t, c0[i] + 0.5*k1c,q,r,f,theta,ua);
                k3t = dt*fu(u0[i] + 0.5*k2t, c0[i] + 0.5*k2c,q,r,f,theta,ua);
                k3c = dt*fc(u0[i] + 0.5*k2t, c0[i] + 0.5*k2c,q,r,f,theta,ua);
                k4t = dt*fu(u0[i] + k3t, c0[i] + k3c,q,r,f,theta,ua);
                k4c = dt*fc(u0[i] + k3t, c0[i] + k3c,q,r,f,theta,ua);

                u1[i] = u0[i] + (k1t + 2.0*k2t + 2.0*k3t + k4t) / 6.0;
                c1[i] = c0[i] + (k1c + 2.0*k2c + 2.0*k3c + k4c) / 6.0;
        }
}

double findError(vector<double> uF,vector<double> uH2,vector<double> cF,vector<double> cH2)
{
	double err=0;
	for (int i=0;i<uF.size();i++)
	{
		err=max(err,abs(uF[i]-uH2[i]));
		err=max(err,abs(cF[i]-cH2[i]));
	}
	return err;
}

int main(int argc,char ** argv)
{
	int N=1000;
	double L=10;
	double dx;
	int size=pow(N+1,2);
	double A=1,sigma=0.5;
	double pi=4.0*atan(1.0);
	double sbuff=0.9; // safety buffer
	double t=0,tmax=1,dt=0.01;
	double ell=1; // centre of vortices
	double T0=1; // period of blinking

	double q=1,r=1,f=2,theta=1,ua=0;
	vector<double> u(size),uF(size),uH1(size),uH2(size);
	vector<double> c(size),cF(size),cH1(size),cH2(size);
	vector<double> x(N+1),y(N+1);

	int nmins=10000;

	for (int i=1;i<argc;i+=2)
	{
		string arg=argv[i];
		if (arg=="nmins")
			nmins=atoi(argv[i+1]);
		else if (arg=="N")
		{
			N=atoi(argv[i+1]);size=pow(N+1,2);
			u.resize(size);c.resize(size);
			x.resize(N+1);y.resize(N+1);
		}
		else if (arg=="L")
			L=atof(argv[i+1]);
		else
		{
			cout << "Parameter " << argv[i] << " not recognised" << endl;
			exit(0);
		}
	}

	dx=1.0*L/N; // allows for both L and N to change above

	for (int i=0;i<=N;i++)
	{
		x[i]=i*dx-L/2;
		for (int j=0;j<=N;j++)
		{
			y[j]=j*dx-L/2;
			u[idx(i,j,N)]=A*exp(-(pow(x[i],2)+pow(y[j],2))/pow(sigma,2))+ua;
			c[idx(i,j,N)]=1.0;
			u[idx(i,j,N)]=1.0;
		}
	}

	cout << av(u) << " " << av(c) << endl;

	string oname="output_new",pname="profile_new";
	for (int i=1;i<argc;i+=2)
	{
		oname+="-"+string(argv[i])+"-"+string(argv[i+1]);
		pname+="-"+string(argv[i])+"-"+string(argv[i+1]);
	}
	oname+=".dat";pname+=".dat";
	cout << oname << " " << pname << endl;
	ofstream out(oname);

	double st=clock();

	while (t<tmax && (clock()-st)/CLOCKS_PER_SEC < sbuff*nmins*60)
	{
		double err,eps=1e-6;
		double xs,ys;	
		if (sin(2*pi*t/T0)<0)
		{
			xs=ell;ys=0;
		}
		else
		{
			xs=-ell;ys=0;
		}
		rk(u,  c  , uF, cF,1.0*dt,q,r,f,theta,ua);
		rk(u,  c  ,uH1,cH1,0.5*dt,q,r,f,theta,ua);
		rk(uH1,cH1,uH2,cH2,0.5*dt,q,r,f,theta,ua);
		err=findError(uF,uH2,cF,cH2);
		if (err<eps)
		{
			u=uH2;c=cH2;t+=dt;
			cout << t << " " << av(u) << " " << av(c) << endl;
		}
		else
		{
			cout << "dt = " << dt << " err = " << err << endl;
		}
		if (err==0)
			dt*=5;
		else
			dt*=clamp(pow(eps/err,0.5),0.2,5.0);
	}

	output(u,c,x,y,N,pname);


}


