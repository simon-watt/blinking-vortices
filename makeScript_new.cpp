#include<iostream>
#include<fstream>
#include<string>

using namespace std;

int main(int argc, char ** argv)
{
	string cmd="time ./chaotic2D_new";
	string fname="subscript_new";

	for (int i=1;i<argc;i+=2)
        {
		cmd+=" "+string(argv[i])+" "+string(argv[i+1]);
		fname+="-"+string(argv[i])+"-"+string(argv[i+1]);
	}
	fname+=".sh";

	cout << cmd << endl;
	cout << fname << endl;

	ofstream out(fname);
	out << "#PBS -l wd" << endl;
	out << "#PBS -q normal" << endl;
	out << "#PBS -l walltime=48:00:00,mem=32Gb,ncpus=48,jobfs=10000MB" << endl;
	out << "#PBS -o /scratch/ad68/sw4657/logs/" << endl;
	out << "#PBS -e /scratch/ad68/sw4657/logs/" << endl;
	out << "export OMP_NUM_THREADS=$PBS_NCPUS" << endl;
	out << "cp chaotic2D_new $PBS_JOBFS" << endl;
	out << "cd $PBS_JOBFS" << endl;
	out << cmd << endl;
	out << "cp output*.dat /scratch/ad68/sw4657/Chaotic/TwoD/ChangingAdvection/" << endl;
	out << "cp profile*.dat /scratch/ad68/sw4657/Chaotic/TwoD/ChangingAdvection/" << endl;
	out << "echo \"Jobname\"" << endl;
	out << "echo $PBS_JOBNAME" << endl;
	out.close();

	cmd="qsub "+fname;
	cout << cmd << endl;
	system(cmd.c_str());
	cmd="cat "+fname;
	cout << cmd << endl;
	system(cmd.c_str());
}

/*
 *
cp chaotic2D_omp $PBS_JOBFS
cd $PBS_JOBFS
time ./chaotic2D_omp Da 20
cp output*.dat /scratch/ad68/sw4657/Chaotic/TwoD/ChangingAdvection/
cp profile*.dat /scratch/ad68/sw4657/Chaotic/TwoD/ChangingAdvection/
*/

