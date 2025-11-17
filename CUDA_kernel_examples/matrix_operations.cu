
#include "cuda_common.cuh"
#include <math.h>

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
    int *d_A, *d_B, *d_out;
    int *h_A, *h_B, *h_out; 
    // Allocate on host (CPU)
    h_A = malloc(sizeof(int) * N);
    h_B = malloc(sizeof(int) * N);
    h_out = malloc(sizeof(int) * N);
    // Initialize arrays
    for (int i = 0; i < N; ++i){
        h_A[i] = 1;
        h_B[i] = 10;
    }
    // Allocate on device (GPU)
    cudaMalloc((void**)&d_A, sizeof(int) * N);
    cudaMalloc((void**)&d_B, sizeof(int) * N);
    cudaMalloc((void**)&d_C, sizeof(int) * N);
    cudaMemcpy(d_A, h_A, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int) * N, cudaMemcpyHostToDevice);

    // Execute kernel on device
    int grid_size = (N + block_size - 1) / block_size;
    vector_add<<grid_size, block_size>>(d_A, d_B, d_out, N);
    // Get result on host
    cudaMemcpy(h_C, d_C, sizeof(int) * N, cudaMemcpyDeviceToHost);
    

    // Deallocate
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(A);
    free(B);
    free(out);
}



int main(){
    test_add(40, 3);
}

