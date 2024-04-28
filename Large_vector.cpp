#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 100000

__global__ void addVectors(float *a, float *b, float *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N)
        c[index] = a[index] + b[index];
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    c = (float*)malloc(N * sizeof(float));

    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_c, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    addVectors<<<numBlocks, blockSize>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}

//0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100 102 104 106 108 110 112 114 116 118 120 122 124 126 128 130 132 134 136 138 140 142 144 146 148 150 152 154 156 158 160 162 164 166 168 170 172 174 176 178 180 182 184 186 188 190 192 194 196 198 200

//Explain 

// Header Includes:
// #include <iostream>: Provides input/output operations.
// #include <cuda_runtime.h>: Provides CUDA runtime APIs.
// #include <device_launch_parameters.h>: Provides CUDA launch parameters.
// Kernel Function:
// __global__ void addVectors(float *a, float *b, float *c): This is the CUDA kernel function responsible for adding corresponding elements of two input arrays a and b and storing the result in the output array c.
// Inside the kernel function, the index of the current thread is calculated using threadIdx.x (thread index within a block) and blockIdx.x (block index within a grid) to access the elements of the arrays.
// The kernel ensures that the index is within the array bounds (N) before performing the addition operation.
// Main Function:
// Memory allocation for host and device arrays:
// float *a, *b, *c;: Host arrays for input and output data.
// float *d_a, *d_b, *d_c;: Device arrays for input and output data.
// Memory allocation on the device using cudaMalloc for storing input and output arrays.
// Initialization of input arrays a and b with values from 0 to N-1.
// Memory copy from host to device using cudaMemcpy to transfer input data (a and b) from host to device.
// Calculation of CUDA grid and block dimensions based on the array size (N) and block size (256).
// Kernel launch using <<<numBlocks, blockSize>>> syntax to execute the addVectors kernel on the GPU.
// Memory copy from device to host using cudaMemcpy to transfer the result (c) from device to host.
// Printing the result array c containing the sum of corresponding elements of a and b.
// Freeing the allocated memory on both host and device using cudaFree and free functions.
// Explanation:
// This CUDA code demonstrates vector addition on the GPU, where each thread computes the sum of corresponding elements of two input arrays (a and b) and stores the result in the output array c.
// The code utilizes CUDA's parallel execution model to distribute the computation across multiple threads running on the GPU.
// Memory management functions (cudaMalloc, cudaMemcpy, cudaFree) are used to manage data transfer between the host (CPU) and the device (GPU).
// Kernel launch syntax (<<<numBlocks, blockSize>>>) specifies the grid and block dimensions for executing the kernel on the GPU.
// Overall, this code efficiently performs vector addition using CUDA parallelism, leveraging the GPU's computational power for accelerating the computation.
