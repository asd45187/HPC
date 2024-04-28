#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 1024

__global__ void matmul(float *A, float *B, float *C, int N) {
    __shared__ float smem[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int i = by * blockDim.y * N + ty * N + bx * blockDim.x + tx;
    float sum = 0;
    for (int k = 0; k < N; k += blockDim.x) {
        if (k + tx < N) {
            smem[ty][tx] = A[i];
            smem[ty][tx + 16] = B[k * N + i];
        }
        __syncthreads();
        if (ty < 16 && tx < 16 && k + ty < N) {
            for (int j = 0; j < 16; j++)
                sum += smem[ty][j] * smem[tx + 16][j];
        }
        __syncthreads();
    }
    if (ty < 16 && tx < 16 && i < N * N)
        C[i] = sum;
}

int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    A = (float*)malloc(N * N * sizeof(float));
    B = (float*)malloc(N * N * sizeof(float));
    C = (float*)malloc(N * N * sizeof(float));

    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        A[i] = i;
        B[i] = i;
    }

    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N * N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}

// 0 1 4 9 16 25 36 49 64 81 100 121 144 169 196 225 256 289 324 361 400 441 484 529 576 625 676 729 784 841 900 961 1024 1089 1156 1225 1296 1369

// Explain 


// 1. **Header Includes**:
//    - `#include <iostream>`: Provides input/output operations.
//    - `#include <cuda_runtime.h>`: Provides CUDA runtime APIs.
//    - `#include <device_launch_parameters.h>`: Provides CUDA launch parameters.

// 2. **Kernel Function**:
//    - `__global__ void matmul(float *A, float *B, float *C, int N)`: This is the CUDA kernel function responsible for performing matrix multiplication.
//    - It utilizes shared memory (`smem`) to store subsets of matrix `A` and `B` for efficient memory access.
//    - Each thread block consists of 256 threads arranged in a 16x16 grid (`threadsPerBlock`) to process small tiles of the input matrices.
//    - The kernel calculates the global index `i` of each thread and performs matrix multiplication using the tiles stored in shared memory.
//    - The result is stored in the output matrix `C`.

// 3. **Main Function**:
//    - Memory allocation for host and device arrays (`A`, `B`, `C` and `d_A`, `d_B`, `d_C`).
//    - Memory allocation on the device using `cudaMalloc` for storing input and output matrices.
//    - Initialization of input matrices `A` and `B` with sequential values.
//    - Memory copy from host to device using `cudaMemcpy` to transfer input matrices (`A` and `B`) from host to device.
//    - Calculation of CUDA grid and block dimensions based on the matrix size (`N`) and block size (`16x16`).
//    - Kernel launch using `<<<numBlocks, threadsPerBlock>>>` syntax to execute the `matmul` kernel on the GPU.
//    - Memory copy from device to host using `cudaMemcpy` to transfer the result matrix (`C`) from device to host.
//    - Printing the result matrix `C`.

// 4. **Output**:
//    - After the matrix multiplication operation is performed on the GPU, the result matrix `C` is printed to the console.

// 5. **Explanation**:
//    - This CUDA code demonstrates matrix multiplication on the GPU using a tiled matrix multiplication algorithm.
//    - The kernel function `matmul` efficiently utilizes shared memory to minimize global memory accesses and improve memory bandwidth utilization.
//    - By launching multiple threads in parallel, CUDA allows for efficient computation of large matrix multiplications on the GPU.
//    - Memory management functions (`cudaMalloc`, `cudaMemcpy`, `cudaFree`) are used to manage data transfer between the host (CPU) and the device (GPU).
//    - Overall, this code efficiently performs matrix multiplication using CUDA parallelism, leveraging the GPU's computational power for accelerating the computation.
