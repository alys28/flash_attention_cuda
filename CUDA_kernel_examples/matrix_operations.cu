#include "cuda_common.cuh"
#include <math.h>

// Simple addition kernel
__global__ void vector_add(int* A, int* B, int* out, int N){
    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){ // Needed because CUDA will launch a number of threads equal to the multiple of the warp size, so potentially more than N.
        out[i] = A[i] + B[i];
    }
}


// Exercise 1
__global__ void matrix_add(int* A, int* B, int* out, int N_x, int N_y){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column < N_x && row < N_y){
        out[row * N_x + column] = A[row * N_x + column] + B[row * N_x + column];
    }
}


// Exercise 2
__global__ void matrix_multiply(int* A, int* B, int* out, int N_A_x, int N_A_y, int N_B_x, int N_B_y){
    // Naive implementation: Each thread is responsible for one (i, j) entry in out (stored in row-major)
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column < N_B_x && row < N_A_y){
        int sum = 0; // Prevent multiple writes to GPU
        for (int i = 0; i < N_A_x; ++i){
            sum += A[row * N_A_x + i] * B[i * N_B_x + column];
        }
        out[row * N_B_x + column] = 0;
    }
}


// BONUS (later): Tiled matmul with shared memory


void test_matrix_multiply(int N_A_x, int N_A_y, int N_B_x, int N_B_y, int block_size_x, int block_size_y){
    int *d_A, *d_B, *d_C;
    int *h_A = malloc(sizeof(int) * N_A_x * N_A_y);
    int *h_B = malloc(sizeof(int) * N_B_x * N_B_y);
    int *h_C = malloc(sizeof(int) * N_B_x * N_A_y);

    for (int i = 0; i < N_A_x * N_A_y; ++i){
        h_A[i] = rand() % 100;
    }

    for (int i = 0; i < N_B_x * N_B_y; ++i){
        h_B[i] = rand() % 100;
    }

    int N_A = N_A_x * N_A_y;
    int N_B = N_B_x * N_B_y;
    int N_C = N_C_x * N_C_y;
    cudaMalloc((void**)&d_A, sizeof(int) * N_A);
    cudaMalloc((void**)&d_B, sizeof(int) * N_B);
    cudaMalloc((void**)&d_C, sizeof(int) * N_C);
    cudaMemcpy(d_A, h_A, sizeof(int) * N_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int) * N_B, cudaMemcpyHostToDevice);
    

    dim3 grid_size((N_C_x + block_size_x - 1) / block_size_x, (N_C_y + block_size_y - 1) / block_size_y, 1);
    dim3 block_size(block_size_x, block_size_y, 1)

    matrix_add<<grid_size, block_size>>(d_A, d_B, d_out, N_C_x, N_C_y);
    cudaMemcpy(h_C, d_C, sizeof(int) * N_C, cudaMemcpyDeviceToHost);

    // Deallocate
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
}



void test_matrix_add(int N_x, int N_y, int block_size_x, int block_size_y){
    int *d_A, *d_B, *d_C;
    int *h_A = malloc(sizeof(int) * N_x * N_y);
    int *h_B = malloc(sizeof(int) * N_x * N_y);
    int *h_C = malloc(sizeof(int) * N_x * N_y);

    for (int i = 0; i < N_x * N_y; ++i){
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }
    int N = N_x * N_y;
    cudaMalloc((void**)&d_A, sizeof(int) * N);
    cudaMalloc((void**)&d_B, sizeof(int) * N);
    cudaMalloc((void**)&d_C, sizeof(int) * N);
    cudaMemcpy(d_A, h_A, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int) * N, cudaMemcpyHostToDevice);
    

    dim3 grid_size = dim3((N_x + block_size_x - 1) / block_size_x, N_y + block_size_y - 1) / block_size_y, 1)
    dim3 block_size = dim3(block_size_x, block_size_y, 1)

    matrix_add<<grid_size, block_size>>(d_A, d_B, d_out, N_x, N_y);
    cudaMemcpy(h_C, d_C, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Deallocate
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
}

void test_vector_add(int N, int block_size){
    int *d_A, *d_B, *d_out;
    int *h_A, *h_B, *h_out; 
    // Allocate on host (CPU)
    h_A = malloc(sizeof(int) * N);
    h_B = malloc(sizeof(int) * N);
    h_out = malloc(sizeof(int) * N);
    // Initialize arrays
    for (int i = 0; i < N; ++i){
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
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
    cudaMemcpy(h_out, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Deallocate
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);

    free(h_A);
    free(h_B);
    free(h_out);
}

int main(){
    test_vector_add(40, 3);
}

