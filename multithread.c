#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>


const int TIMES = 2;
const int THREAD_NUM = TIMES * TIMES;

struct sub_task
{
	float **A;
	float **B;
	float **C;
	int n;
	int m;
	int p;
	int r_id; // 0, 1, 2,..., TIMES
	int l_id; // 0, 1, 2,..., TIMES
};

void *thread(void *arg)
{
	struct sub_task *task = (struct sub_task *)arg;
	float **A = task->A;
	float **B = task->B;
	float **C = task->C;
	int n = task->n;
	int m = task->m;
	int p = task->p;
	int r_id = task->r_id;
	int l_id = task->l_id;

	int i_right = (r_id + 1) * n / TIMES, j_right = (l_id + 1) * p / TIMES;
	for (int i = r_id * n / TIMES; i < i_right; i++)
		for (int j = l_id * p / TIMES; j < j_right; j++)
		{
			C[i][j] = 0;
			for (int k = 0; k < m; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
		}

	return NULL;
}

void *thread_maxv_line_first(void *arg)
{
	struct sub_task *task = (struct sub_task *)arg;
	float **A = task->A;
	float **B = task->B;
	float **C = task->C;
	int N = task->n;
	int M = task->m;
	int P = task->p;
	int r_id = task->r_id;
	// int l_id = task->l_id;

	int i_right = (r_id + 1) * N / THREAD_NUM;
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
		ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
	float scratchpad[8];
	int col_reduced = M - M % 64;
	int col_reduced_32 = M - M % 32;

	for (int i = r_id * N / THREAD_NUM; i < i_right; i++)
	{
		for (int p = 0; p < P; p++)
		{
			float res = 0;
			for (int j = 0; j < col_reduced; j += 64)
			{
				// ymm 8 - 15 装载 B[p] 列
				ymm8 = __builtin_ia32_loadups256(&B[p][j]);
				ymm9 = __builtin_ia32_loadups256(&B[p][j + 8]);
				ymm10 = __builtin_ia32_loadups256(&B[p][j + 16]);
				ymm11 = __builtin_ia32_loadups256(&B[p][j + 24]);
				ymm12 = __builtin_ia32_loadups256(&B[p][j + 32]);
				ymm13 = __builtin_ia32_loadups256(&B[p][j + 40]);
				ymm14 = __builtin_ia32_loadups256(&B[p][j + 48]);
				ymm15 = __builtin_ia32_loadups256(&B[p][j + 56]);
				// ymm 0 - 7 装载 A[i] 行
				ymm0 = __builtin_ia32_loadups256(&A[i][j]);
				ymm1 = __builtin_ia32_loadups256(&A[i][j + 8]);
				ymm2 = __builtin_ia32_loadups256(&A[i][j + 16]);
				ymm3 = __builtin_ia32_loadups256(&A[i][j + 24]);
				ymm4 = __builtin_ia32_loadups256(&A[i][j + 32]);
				ymm5 = __builtin_ia32_loadups256(&A[i][j + 40]);
				ymm6 = __builtin_ia32_loadups256(&A[i][j + 48]);
				ymm7 = __builtin_ia32_loadups256(&A[i][j + 56]);
				// A[i][] * B[][p]
				ymm0 = __builtin_ia32_mulps256(ymm0, ymm8);
				ymm1 = __builtin_ia32_mulps256(ymm1, ymm9);
				ymm2 = __builtin_ia32_mulps256(ymm2, ymm10);
				ymm3 = __builtin_ia32_mulps256(ymm3, ymm11);
				ymm4 = __builtin_ia32_mulps256(ymm4, ymm12);
				ymm5 = __builtin_ia32_mulps256(ymm5, ymm13);
				ymm6 = __builtin_ia32_mulps256(ymm6, ymm14);
				ymm7 = __builtin_ia32_mulps256(ymm7, ymm15);
				// 对 ymm0-ymm7求和得结果
				ymm0 = __builtin_ia32_addps256(ymm0, ymm1);
				ymm2 = __builtin_ia32_addps256(ymm2, ymm3);
				ymm4 = __builtin_ia32_addps256(ymm4, ymm5);
				ymm6 = __builtin_ia32_addps256(ymm6, ymm7);
				ymm0 = __builtin_ia32_addps256(ymm0, ymm2);
				ymm4 = __builtin_ia32_addps256(ymm4, ymm6);
				ymm0 = __builtin_ia32_addps256(ymm0, ymm4);

				__builtin_ia32_storeups256(scratchpad, ymm0);
				for (int k = 0; k < 8; k++)
					res += scratchpad[k];
			}
			
			for (int j = col_reduced; j < col_reduced_32; j += 32)
			{
				ymm8 = __builtin_ia32_loadups256(&B[p][j]);
				ymm9 = __builtin_ia32_loadups256(&B[p][j + 8]);
				ymm10 = __builtin_ia32_loadups256(&B[p][j + 16]);
				ymm11 = __builtin_ia32_loadups256(&B[p][j + 24]);

				ymm0 = __builtin_ia32_loadups256(&A[i][j]);
				ymm1 = __builtin_ia32_loadups256(&A[i][j + 8]);
				ymm2 = __builtin_ia32_loadups256(&A[i][j + 16]);
				ymm3 = __builtin_ia32_loadups256(&A[i][j + 24]);

				ymm0 = __builtin_ia32_mulps256(ymm0, ymm8);
				ymm1 = __builtin_ia32_mulps256(ymm1, ymm9);
				ymm2 = __builtin_ia32_mulps256(ymm2, ymm10);
				ymm3 = __builtin_ia32_mulps256(ymm3, ymm11);

				ymm0 = __builtin_ia32_addps256(ymm0, ymm1);
				ymm2 = __builtin_ia32_addps256(ymm2, ymm3);
				ymm0 = __builtin_ia32_addps256(ymm0, ymm2);

				__builtin_ia32_storeups256(scratchpad, ymm0);
				for (int k = 0; k < 8; k++)
					res += scratchpad[k];
			}

			for (int l = col_reduced_32; l < M; l++)
			{
				res += A[i][l] * B[p][l];
			}
			C[i][p] = res;
		}
	}

	return NULL;
}

// 传入的第二个矩阵应该先转置
void matrix_multiply_maxv_line_first(float **A, float **B, float **C, int N, int M, int P)
{
	pthread_t threads[THREAD_NUM];
	struct sub_task tasks[THREAD_NUM];
	for (int i = 0; i < THREAD_NUM; i++)
	{
		tasks[i].r_id = i;
		// tasks[i][j].l_id = j;
		tasks[i].A = A;
		tasks[i].B = B;
		tasks[i].C = C;
		tasks[i].n = N;
		tasks[i].m = M;
		tasks[i].p = P;
		pthread_create(&threads[i], NULL, thread_maxv_line_first, &tasks[i]);
	}

	for (int i = 0; i < THREAD_NUM; i++)
		pthread_join(threads[i], NULL);
}

void *thread_line_first(void *arg)
{
	struct sub_task *task = (struct sub_task *)arg;
	float **A = task->A;
	float **B = task->B;
	float **C = task->C;
	int n = task->n;
	int m = task->m;
	int p = task->p;
	int r_id = task->r_id;
	int l_id = task->l_id;

	int i_right = (r_id + 1) * n / TIMES, j_right = (l_id + 1) * p / TIMES;
	for (int i = r_id * n / TIMES; i < i_right; i++)
		for (int j = l_id * p / TIMES; j < j_right; j++)
		{
			C[i][j] = 0;
			for (int k = 0; k < m; k++)
			{
				C[i][j] += A[i][k] * B[j][k];
			}
		}

	return NULL;
}

void matrix_multiply(float **A, float **B, float **C, int n, int m, int p)
{
	pthread_t threads[TIMES][TIMES];
	struct sub_task tasks[TIMES][TIMES];
	for (int i = 0; i < TIMES; i++)
	{
		for (int j = 0; j < TIMES; ++j)
		{
			tasks[i][j].r_id = i;
			tasks[i][j].l_id = j;
			tasks[i][j].A = A;
			tasks[i][j].B = B;
			tasks[i][j].C = C;
			tasks[i][j].n = n;
			tasks[i][j].m = m;
			tasks[i][j].p = p;
			pthread_create(&threads[i][j], NULL, thread, &tasks[i][j]);
		}
	}

	for (int i = 0; i < TIMES; i++)
		for (int j = 0; j < TIMES; ++j)
		{
			pthread_join(threads[i][j], NULL);
		}
}

void matrix_multiply_line_first(float **A, float **B, float **C, int n, int m, int p)
{
	pthread_t threads[TIMES][TIMES];
	struct sub_task tasks[TIMES][TIMES];
	for (int i = 0; i < TIMES; i++)
	{
		for (int j = 0; j < TIMES; ++j)
		{
			tasks[i][j].r_id = i;
			tasks[i][j].l_id = j;
			tasks[i][j].A = A;
			tasks[i][j].B = B;
			tasks[i][j].C = C;
			tasks[i][j].n = n;
			tasks[i][j].m = m;
			tasks[i][j].p = p;
			pthread_create(&threads[i][j], NULL, thread_line_first, &tasks[i][j]);
		}
	}

	for (int i = 0; i < TIMES; i++)
		for (int j = 0; j < TIMES; ++j)
		{
			pthread_join(threads[i][j], NULL);
		}
}

void matrix_print(float **M, int n, int m)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			printf("%f ", M[i][j]);
		}
		putchar('\n');
	}
}

float **matrix_malloc(int n, int m)
{
	float **M;
	M = (float **)malloc(sizeof(float *) * n);
	for (int i = 0; i < n; ++i)
	{
		M[i] = (float *)malloc(sizeof(float) * m);
	}
	return M;
}

void matrix_init(float **M, int n, int m)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			M[i][j] = rand() % 10;
		}
	}
}

float **matrix_transpose(float **M, int n, int m)
{
	float **T = matrix_malloc(m, n);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			T[j][i] = M[i][j];
	return T;
}

int main(int argc, char *argv[])
{
	int n, m, p;
	if (argc == 2)
	{
		n = m = p = atoi(argv[1]);
	}
	else if (argc == 4)
	{
		n = atoi(argv[1]);
		m = atoi(argv[2]);
		p = atoi(argv[3]);
	}
	else
	{
		printf("Help:\n");
		printf("./multithread n\n");
		printf("./multithread n m p\n");
		return 0;
	}

	float **A = matrix_malloc(n, m);
	if (!A)
	{
		perror("matrix_malloc()");
	}
	float **B = matrix_malloc(m, p);
	if (!B)
	{
		perror("matrix_malloc()");
	}
	float **C = matrix_malloc(n, p);
	if (!C)
	{
		perror("matrix_malloc()");
	}

	matrix_init(A, n, m);
	matrix_init(B, m, p);

	//	matrix_print(A, n, m);
	//	matrix_print(B, m, p);
	//	matrix_print(T, p, m);

	struct timespec t1, t2, t3, t4, t5, t6;

	clock_gettime(CLOCK_MONOTONIC, &t1);
	matrix_multiply(A, B, C, n, m, p);
	clock_gettime(CLOCK_MONOTONIC, &t2);
	printf("4线程算法用时：%lf\n", t2.tv_sec - t1.tv_sec + (t2.tv_nsec - t1.tv_nsec) / 1000000000.0);
	matrix_print(C, n, p);

	clock_gettime(CLOCK_MONOTONIC, &t3);
	float **T = matrix_transpose(B, m, p);
	if (!T)
	{
		perror("matrix_transpose()");
	}
	matrix_multiply_line_first(A, T, C, n, m, p);
	clock_gettime(CLOCK_MONOTONIC, &t4);

	printf("4线程+行优先用时：%lf\n", t4.tv_sec - t3.tv_sec + (t4.tv_nsec - t3.tv_nsec) / 1000000000.0);
	matrix_print(C, n, p);

	clock_gettime(CLOCK_MONOTONIC, &t5);
	T = matrix_transpose(B, m, p);
	if (!T)
	{
		perror("matrix_transpose()");
	}
	matrix_multiply_maxv_line_first(A, T, C, n, m, p);
	clock_gettime(CLOCK_MONOTONIC, &t6);

	printf("行优先+向量指令+多线程用时：%lf\n", t6.tv_sec - t5.tv_sec + (t6.tv_nsec - t5.tv_nsec) / 1000000000.0);
	matrix_print(C, n, p);
}
