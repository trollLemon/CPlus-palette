#include <iostream>

// CUDA kernel to add two arrays on the GPU
__global__ void addArrays(int* a, int* b, int* c, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}
