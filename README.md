# PCA-Simple-warp-divergence---Implement-Sum-Reduction.
Refer to the kernel reduceUnrolling8 and implement the kernel reduceUnrolling16, in which each thread handles 16 data blocks. Compare kernel performance with reduceUnrolling8 and use the proper metrics and events with nvprof to explain any difference in performance.

## Aim:
To compare the performance of the reduceUnrolling16 kernel with reduceUnrolling8 by implementing and profiling them using nvprof and analyze the performance variations to compare the two kernels through the collected metrics and events.



## Procedure:Step 1:

Step 1:
Import the necessary files and libraries.
Step 2:
Apply the Interleaved Pair Approach by implementing it recursively.
Step 3:
Implement the Interleaved Pair Approach with less divergence using a CUDA kernel function.
Step 4:
Apply optimizations, such as loop unrolling or memory access patterns, to reduce divergence in the CUDA kernel function.
Step 5:
Implement specific optimizations, such as unrolling warps or using a specific block size, in the CUDA kernel function.
Step 6: 
Declare the main method/function. Set up the device, initialize the size and block size, allocate host memory and device memory, and call the CUDA kernels declared in the functions.
Step 7:
Implement error validation and handling for CUDA function executions.
Step 8:
Free the host and device memory, reset the device, and check the results.

## program:
Developed By: Thenmozhi P
Register No.: 212221230116
``  ````
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This code implements the interleaved and neighbor-paired approaches to
 * parallel reduction in CUDA. For this example, the sum operation is used. A
 * variety of optimizations on parallel reduction aimed at reducing divergence
 * are also demonstrated, such as unrolling.
 */
````
// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int *data, int const size)
{
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

// Neighbored Pair Implementation with divergence
_global_ void reduceNeighbored (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}```

// Neighbored Pair Implementation with less divergence
_global_ void reduceNeighboredLess (int *g_idata, int *g_odata,
                                      unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // convert tid into local array index
        int index = 2 * stride * tid;

        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
```
// Interleaved Pair Implementation with less divergence
_global_ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

_global_ void reduceUnrolling2 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
``
_global_ void reduceUnrolling4 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

_global_ void reduceUnrolling8 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
`
_global_ void reduceUnrollWarps8 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

_global_ void reduceCompleteUnrollWarps8 (int *g_idata, int *g_odata,
        unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
_global_ void reduceCompleteUnroll(int *g_idata, int *g_odata,
                                     unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  idata[tid] += idata[tid + 256];

    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  idata[tid] += idata[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

_global_ void reduceUnrollWarps (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling last warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size

    if(argc > 1)
    {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)( rand() & 0xFF );
    }

    memcpy (tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    // cpu reduction
    iStart = seconds();
    int cpu_sum = recursiveReduce (tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1: reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighbored with less divergence
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 4: reduceUnrolling2
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling2  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 2, block.x);

    // kernel 5: reduceUnrolling4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling4  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 4, block.x);

    // kernel 6: reduceUnrolling8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling8  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    for (int i = 0; i < grid.x / 16; i++) gpu_sum += h_odata[i];

    // kernel 8: reduceUnrollWarps8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu UnrollWarp8 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);


    // kernel 9: reduceCompleteUnrollWarsp8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll8  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 9: reduceCompleteUnroll
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    switch (blocksize)
    {
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 512:
        reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 256:
        reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 128:
        reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 64:
        reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    }

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll   elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}
````
## Output:

![image](https://github.com/Thenmozhi-Palanisamy/PCA-Simple-warp-divergence---Implement-Sum-Reduction./assets/95198708/4eb6052c-1e4e-4226-af4e-04f5f566981b)

Kernel reduceUnrolling16:
``
root@MidPC:/home/student/Desktop# nvcc first.cu
root@MidPC:/home/student/Desktop# ./a.out
./a.out starting reduction at device 0: NVIDIA GeForce GTX 1660 SUPER     with array size 16777216  grid 32768 block 512
cpu reduce      elapsed 0.033168 sec cpu_sum: 2139353471
gpu Neighbored  elapsed 0.002431 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Neighbored2 elapsed 0.001394 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Interleaved elapsed 0.001289 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Unrolling2  elapsed 0.000802 sec gpu_sum: 2139353471 <<<grid 16384 block 512>>>
gpu Unrolling4  elapsed 0.000514 sec gpu_sum: 2139353471 <<<grid 8192 block 512>>>
gpu Unrolling8  elapsed 0.000355 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Unrolling16 elapsed 0.000337 sec gpu_sum: 2139353471 <<<grid 2048 block 512>>>
gpu UnrollWarp8 elapsed 0.000373 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll8  elapsed 0.000368 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll   elapsed 0.000369 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
root@MidPC:/home/student/Desktop# nvprof ./a.out
==18570== NVPROF is profiling process 18570, command: ./a.out
./a.out starting reduction at device 0: NVIDIA GeForce GTX 1660 SUPER     with array size 16777216  grid 32768 block 512
cpu reduce      elapsed 0.032530 sec cpu_sum: 2139353471
gpu Neighbored  elapsed 0.002478 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Neighbored2 elapsed 0.001474 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Interleaved elapsed 0.001343 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Unrolling2  elapsed 0.000831 sec gpu_sum: 2139353471 <<<grid 16384 block 512>>>
gpu Unrolling4  elapsed 0.000493 sec gpu_sum: 2139353471 <<<grid 8192 block 512>>>
gpu Unrolling8  elapsed 0.000291 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Unrolling16 elapsed 0.000345 sec gpu_sum: 2139353471 <<<grid 2048 block 512>>>
gpu UnrollWarp8 elapsed 0.000315 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll8  elapsed 0.000385 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll   elapsed 0.000414 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
==18570== Profiling application: ./a.out
==18570== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   88.24%  58.669ms        10  5.8669ms  5.7803ms  6.3740ms  [CUDA memcpy HtoD]
                    3.68%  2.4462ms         1  2.4462ms  2.4462ms  2.4462ms  reduceNeighbored(int*, int*, unsigned int)
                    2.18%  1.4493ms         1  1.4493ms  1.4493ms  1.4493ms  reduceNeighboredLess(int*, int*, unsigned int)
                    1.98%  1.3188ms         1  1.3188ms  1.3188ms  1.3188ms  reduceInterleaved(int*, int*, unsigned int)
                    1.11%  738.34us         1  738.34us  738.34us  738.34us  reduceUnrolling2(int*, int*, unsigned int)
                    0.60%  398.56us         1  398.56us  398.56us  398.56us  reduceUnrolling4(int*, int*, unsigned int)
                    0.48%  318.82us         1  318.82us  318.82us  318.82us  void reduceCompleteUnroll<unsigned int=512>(int*, int*, unsigned int)
                    0.44%  290.56us         1  290.56us  290.56us  290.56us  reduceCompleteUnrollWarps8(int*, int*, unsigned int)
                    0.44%  289.79us         1  289.79us  289.79us  289.79us  reduceUnrollWarps8(int*, int*, unsigned int)
                    0.40%  266.63us         1  266.63us  266.63us  266.63us  reduceUnrolling8(int*, int*, unsigned int)
                    0.38%  250.88us         1  250.88us  250.88us  250.88us  reduceUnrolling16(int*, int*, unsigned int)
                    0.08%  54.304us        10  5.4300us  1.8880us  11.104us  [CUDA memcpy DtoH]
      API calls:   52.55%  121.32ms         2  60.662ms  72.660us  121.25ms  cudaMalloc
                   25.57%  59.030ms        20  2.9515ms  18.820us  6.4238ms  cudaMemcpy
                   17.57%  40.567ms         1  40.567ms  40.567ms  40.567ms  cudaDeviceReset
                    3.85%  8.8956ms        20  444.78us  75.680us  2.4467ms  cudaDeviceSynchronize
                    0.10%  225.81us        10  22.581us  20.700us  29.160us  cudaLaunchKernel
                    0.10%  225.49us         2  112.75us  47.580us  177.91us  cudaFree
                    0.09%  209.05us        97  2.1550us     170ns  88.790us  cuDeviceGetAttribute
                    0.08%  187.99us         1  187.99us  187.99us  187.99us  cuDeviceTotalMem
                    0.08%  185.81us         1  185.81us  185.81us  185.81us  cudaGetDeviceProperties
                    0.01%  26.890us         1  26.890us  26.890us  26.890us  cuDeviceGetName
                    0.00%  3.9100us         1  3.9100us  3.9100us  3.9100us  cuDeviceGetPCIBusId
                    0.00%  2.4100us         1  2.4100us  2.4100us  2.4100us  cudaSetDevice
                    0.00%  1.2400us         3     413ns     190ns     830ns  cuDeviceGetCount
                    0.00%     650ns         2     325ns     180ns     470ns  cuDeviceGet
                    0.00%     240ns         1     240ns     240ns     240ns  cuDeviceGetUuid
              
 # output:
![image](https://github.com/Thenmozhi-Palanisamy/PCA-Simple-warp-divergence---Implement-Sum-Reduction./assets/95198708/a4471106-db1e-4f04-a7fb-5d6447689
## Result:
The performance of the reduceUnrolling16 kernel has been compared and analyzed with that of the reduceUnrolling8 kernel using nvprof based on the collected metrics and events.

