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
