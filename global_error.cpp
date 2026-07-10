#include<iostream>
#include<fstream>

using namespace std;

int main(int argc, char ** argv)
{
	ifstream in1(argv[1]),in2(argv[2]);
	double error=0;
	double x,y,u1,c1,u2,c2;

	in1 >> x >> y >> u1 >> c1;
	while (!in1.eof() && !in2.eof())
	{
		in2 >> x >> y >> u2 >> c2;
		error+=abs(u1-u2);
		in1 >> x >> y >> u1 >> c1;
	}
	in1.close();
	in2.close();

	cout << "file 1 = " << argv[1] << endl;
	cout << "file 2 = " << argv[2] << endl;
	printf("global error = %e\n",error);
}
