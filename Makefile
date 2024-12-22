all: singlethread multithread cuda
singlethread:singlethread.c
	gcc -mavx -fno-tree-vectorize singlethread.c -O0  -o singlethread
multithread:multithread.c
	gcc -mavx -fno-tree-vectorize multithread.c -O0 -o multithread
cuda:cuda.cu
	nvcc -O0 cuda.cu -o cuda
clean:
	rm -rf singlethread multithread cuda

