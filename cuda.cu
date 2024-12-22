#include <stdio.h>
#include <stdio.h>
#include <time.h>

#define N 5
#define BLOCK 64
float a[N * N], b[N * N], c[N * N];

__device__ float sum(float *cache, int id)
{
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (id < i)
		{
			cache[id] += cache[id + i];
		}
		__syncthreads();
		i /= 2;
	}
	return cache[0];
}

__global__ void matrix_dot(float *a, float *b, float *c)
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	__shared__ float cache[BLOCK];
	int t = threadIdx.x;
	if (t < N)
		cache[t] = a[i * N + t] * b[t * N + j];
	else
		cache[t] = 0;
	__syncthreads();
	sum(cache, t);
	__syncthreads();
	c[i * N + j] = cache[0];
}

__host__ void dot(float *a, float *b, float *c)
{
	float *a_cuda, *b_cuda, *c_cuda;
	cudaMalloc((void **)&a_cuda, N * N * sizeof(float));
	cudaMalloc((void **)&b_cuda, N * N * sizeof(float));
	cudaMalloc((void **)&c_cuda, N * N * sizeof(float));
	cudaMemcpy(a_cuda, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_cuda, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
	dim3 matrix(N, N);
	matrix_dot<<<matrix, BLOCK>>>(a_cuda, b_cuda, c_cuda);
	cudaMemcpy(c, c_cuda, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(a_cuda);
	cudaFree(b_cuda);
	cudaFree(c_cuda);
}

int main()
{
	clock_t startTime, endTime;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a[i * N + j] = rand() % 10;
		}
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			b[i * N + j] = rand() % 10;
		}
	}
	startTime = clock();
	dot(a, b, c);
	endTime = clock();
	printf("GPU加速矩阵乘法用时: %lf\n", 1.0 * (endTime - startTime) / CLOCKS_PER_SEC);
	// printf("A:\n");
	// for (int i = 0; i < N; i++) {
	// 	for (int j = 0; j < N; j++) {
	// 		printf("%f ", a[i*N + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("B:\n");
	// for (int i = 0; i < N; i++) {
	// 	for (int j = 0; j < N; j++) {
	// 		printf("%f ", b[i*N + j]);
	// 	}
	// 	printf("\n");
	// }
	printf("C:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%f ", c[i*N + j]);
		}
		printf("\n");
	}
	return 0;
}