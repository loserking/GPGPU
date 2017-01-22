#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}
//Convert all characters to be '!'
__global__ void SomeTransform(char *input_gpu, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < fsize and input_gpu[idx] != '\n') {
		input_gpu[idx] = '!';
	}
}
//Convert all characters to be capital
__global__ void SomeTransform_1(char *input_gpu, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < fsize and 97 <= input_gpu[idx] and input_gpu[idx] <= 122) {
		input_gpu[idx] = input_gpu[idx] - 32;
	}
}
//Swap all pairs in all words
__global__ void SomeTransform_2(char *input_gpu, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	char temp;
	if (idx < fsize and input_gpu[idx] != '\n' and input_gpu[idx+1] != '\n') {
		temp = input_gpu[idx];
		input_gpu[idx] = input_gpu[idx+1];
		input_gpu[idx+1] = temp;
		idx = idx + 2;
	}
	else if(idx < fsize and input_gpu[idx] != '\n' and input_gpu[idx+1] == '\n')
	{
		input_gpu[idx] = input_gpu[idx];
	}
}
int main(int argc, char **argv)
{
	// init, and check
	if (argc != 2) {
		printf("Usage %s <input text file>\n", argv[0]);
		abort();
	}
	FILE *fp = fopen(argv[1], "r");
	if (not fp) {
		printf("Cannot open %s", argv[1]);
		abort();
	}
	// get file size
	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// read files
	MemoryBuffer<char> text(fsize+1);
	auto text_smem = text.CreateSync(fsize);
	CHECK;
	fread(text_smem.get_cpu_wo(), 1, fsize, fp);
	text_smem.get_cpu_wo()[fsize] = '\0';
	fclose(fp);

	// TODO: do your transform here
	char *input_gpu = text_smem.get_gpu_rw();
	// An example: transform the first 64 characters to '!'
	// Don't transform over the tail
	// And don't transform the line breaks
	SomeTransform<<<8, 32>>>(input_gpu, fsize);
	SomeTransform_1<<<8, 32>>>(input_gpu, fsize);
	SomeTransform_2<<<8, 32>>>(input_gpu, fsize);

	puts(text_smem.get_cpu_ro());
	return 0;
}