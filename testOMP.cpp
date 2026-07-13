#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
#pragma omp parallel
#pragma omp critical
	cout << "hello from " << omp_get_thread_num() << endl;
}

