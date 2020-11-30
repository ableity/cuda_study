#include <iostream>  
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
using namespace std;
__global__ void add(int a, int b, int *c)
{
	*c = a + b;
}
int main(){
	int c;
	int *dev_c;
	cudaMalloc((void**)&dev_c, sizeof(int));
	add << <1, 1 >> >(5, 9, dev_c);
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	cout << "5 + 9 = " << c << endl;
	cudaFree(dev_c);

	system("pause");
}