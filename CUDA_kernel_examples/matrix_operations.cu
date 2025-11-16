
#include "cuda_common.cuh"

// Simple addition kernel
__global__ vector_add(int* A, int* B, int* out, int N){
    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){ // Needed because CUDA will launch a number of threads equal to the multiple of the warp size, so potentially more than N.
        out[i] = A[i] + B[i];
    }
}


// Exercise 1
__global__ matrix_add(int* A, int* B, int* out, int N){

}


// Exercise 2
__global__ matrix_multiply(int* A, int* B, int* out, int N){

}


// BONUS: Tiled matmul with shared memory


void test_add(int N, int block_size){
    int *A, *B, *out;
    // Allocate on host (CPU)
    A = malloc(sizeof(int), N);
    B = malloc(sizeof(int), N);
}


