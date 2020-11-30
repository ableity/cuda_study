#include <iostream>  
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
//#pragma comment(lib,"cudart.lib")
#include "G:\cuda\bitmap\common\cpu_bitmap.h"
#include "G:\cuda\bitmap\common\book.h"
using namespace std;


//__global__ void addKernel(int **C, int **A, int )
//{
//	int idx = threadIdx.x + blockDim.x * blockIdx.x;
//	int idy = threadIdx.y + blockDim.y * blockIdx.y;
//	if (idx < Col && idy < Row) {
//		C[idy][idx] = A[idy][idx] + 10;
//	}
//
//
//void main()
//{
//	float a[10][9];
//	float b[9][11];
//	
//
//	const int h = sizeof(a) / sizeof(a[0]);
//	const int l = sizeof(b[0])/sizeof(float);
//	const int t = sizeof(a[0]) / sizeof(float);
//	int t2 = sizeof(b) / sizeof(b[0]);
//
//	if (t != t2)
//	{
//		cout << "error"<<endl;
//		system("pause");
//	}
//
//	float c[h][l];
//
//	float (*dev_a)[t], (*dev_b)[l], (*dev_c)[l];
//
//	cudaMalloc((void**)&dev_a, sizeof(float **)*t);
//	cudaMalloc((void**)&dev_b, sizeof(float **)*l);
//	cudaMalloc((void**)&dev_c, sizeof(float **)*l);
//
//	for (int i = 0; i < 10; i++)
//	{
//		for (int j = 0; j < 9; j++)
//		{
//			a[i][j] = 1;
//		}
//	}
//	for (int i = 0; i < 9; i++)
//	{
//		for (int j = 0; j < 11; j++)
//		{
//			b[i][j] = 2;
//		}
//	}
//
//	cudaMemcpy(dev_a, a, sizeof(float*) * t, cudaMemcpyHostToDevice);
//	cudaMemcpy(dev_b, b, sizeof(float*) * l, cudaMemcpyHostToDevice);
//	cudaMemcpy(dev_c, c, sizeof(float*) * l, cudaMemcpyHostToDevice);
//
//
//
//
//	cout << h<<endl;
//	cout << l<<endl;
//
//	system("pause");
//}

#define imin(a,b) (a<b?a:b)

//const int N = 10000;
const int threadsPerBlock = 256;



__global__ void dot(float *a, float *b, float *c, int *n) {
	__shared__ float cache[threadsPerBlock];  //每个block一个共享内存
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	int NT = *n;

	float   temp = 0;
	while (tid < NT) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	// set the cache values
	cache[cacheIndex] = temp; //对于越界的的索引，初始化为0

	// synchronize threads in this block
	__syncthreads();

	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}


float vectordot(float *a, float *b, int	N)
{
	int blocksPerGrid =
		imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);//最多开到32个block
	float   c, *partial_c;
	float   *dev_a, *dev_b, *dev_partial_c;
	int *n, *n_temp;

	// allocate memory on the cpu side
	n_temp = (int*)malloc(sizeof(int));
	partial_c = (float*)malloc(blocksPerGrid*sizeof(float));//每个block分一个空

	// allocate the memory on the GPU
	HANDLE_ERROR(cudaMalloc((void**)&dev_a,
		N*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b,
		N*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c,
		blocksPerGrid*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&n, sizeof(int)));

	// copy the arrays 'a' and 'b' to the GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(float),
		cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(float),
		cudaMemcpyHostToDevice));
	*n_temp = N;
	HANDLE_ERROR(cudaMemcpy(n, n_temp, sizeof(int),
		cudaMemcpyHostToDevice));

	dot << <blocksPerGrid, threadsPerBlock >> >(dev_a, dev_b,
		dev_partial_c, n);

	// copy the array 'c' back from the GPU to the CPU
	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c,
		blocksPerGrid*sizeof(float),
		cudaMemcpyDeviceToHost));

	// finish up on the CPU side
	c = 0;
	for (int i = 0; i<blocksPerGrid; i++) {
		c += partial_c[i];
	}

	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_partial_c));

	// free memory on the cpu side
	free(partial_c);
	return c;
}


int main(void) {
	int N = 1000;
	int blocksPerGrid =
		imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);//最多开到32个block


	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Number of SMs:%d\n",prop.multiProcessorCount);


	float   *a, *b;

	// allocate memory on the cpu side
	a = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float));

	// fill in the host memory with data
	for (int i = 0; i<N; i++) {
		a[i] = 1;
		b[i] = 1;
	}
	printf("sum is %f", vectordot(a,b,N));

	system("pause");
}

//#define dim 1000
//
//struct cucomplex{
//	float r;
//	float i;
//	__device__ cucomplex(float a, float b) : r(a), i(b){}
//	__device__ float magnitude2(void)
//	{
//		return r*r + i*i;
//	}
//
//	__device__ cucomplex operator*(const cucomplex& a)
//
//	{
//		return cucomplex(r*a.r - i*a.i, i*a.r + r*a.i);
//	}
//
//	__device__ cucomplex operator+(const cucomplex& a)
//	{
//		return cucomplex(r + a.r, i + a.i);
//	}
//};
//
//
//
//
//__device__ int julia(int x, int y)
//{
//	const float scale = 1.5;
//	float jx = scale * (float)(dim / 2 - x) / (dim / 2);
//	float jy = scale * (float)(dim / 2 - y) / (dim / 2);
//
//	cucomplex c(-0.8, 0.156);
//	cucomplex a(jx, jy);
//	int i = 0;
//	for (i = 0; i < 200; i++)
//	{
//		a = a*a + c;
//		if (a.magnitude2() > 1000)
//			return 0;
//	}
//	return 1;
//}
//
//__global__ void kernel(unsigned char *ptr)
//{
//	int x = blockIdx.x;
//	int y = blockIdx.y;
//
//	int offset = x + y * gridDim.x;
//
//	int juliavalue = julia(x, y);
//	ptr[offset * 4 + 0] = 255 * juliavalue;
//	ptr[offset * 4 + 1] = 0;
//	ptr[offset * 4 + 2] = 0;
//	ptr[offset * 4 + 3] = 255;
//}
//
//
//
//int main()
//{
//	CPUBitmap bitmap(dim, dim);
//	unsigned char *dev_bitmap;
//	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
//	dim3 grid(dim, dim);
//	kernel << <grid, 1 >> >(dev_bitmap);
//	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
//
//	bitmap.display_and_exit();
//	cudaFree(dev_bitmap);
//	return 0;
//}

//__global__ void add(int a, int b, int *c)
//{
//	*c = a + b;
//}
//int main(){
//	int c;
//	int *dev_c;
//	cudaMalloc((void**)&dev_c, sizeof(int));
//	add << <1, 1 >> >(5, 9, dev_c);
//	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
//	cout << "5 + 9 = " << c << endl;
//	cudaFree(dev_c);
//
//	system("pause");
//}