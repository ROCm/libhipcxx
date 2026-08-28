//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef LIBCUDACXX_FORCE_INCLUDE_HIP
#define LIBCUDACXX_FORCE_INCLUDE_HIP

#include "cuda_runtime.h"
// TODO(HIP/AMD): this is a temporary WAR to create leass file modifications.
// This should be only in the test_macros.h. Unfortunately many tests do not
// include this header.
#ifndef NV_IF_TARGET
#define NV_IF_TARGET NV_IF_TARGET_LIBHIPCXX
#endif
#ifndef NV_IS_HOST
#define NV_IS_HOST NV_IS_HOST_LIBHIPCXX
#endif
#ifndef NV_IS_DEVICE
#define NV_IS_DEVICE NV_IS_DEVICE_LIBHIPCXX
#endif

// We use <stdio.h> instead of <iostream> to avoid relying on the host system's
// C++ standard library.
#include <stdio.h>
#include <stdlib.h>

#define HIP_CALL(err, ...) \
    do { \
        err = __VA_ARGS__; \
        if (err != cudaSuccess) \
        { \
            printf("HIP ERROR, line %d: %s: %s\n", __LINE__,\
                   cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (false)

#define CUDA_CALL HIP_CALL

void list_devices()
{
    cudaError_t err;
    int device_count;
    HIP_CALL(err, cudaGetDeviceCount(&device_count));
    printf("HIP devices found: %d\n", device_count);

    int selected_device;
    HIP_CALL(err, cudaGetDevice(&selected_device));

    for (int dev = 0; dev < device_count; ++dev)
    {
        cudaDeviceProp device_prop;
        HIP_CALL(err, cudaGetDeviceProperties(&device_prop, dev));

        printf("Device %d: \"%s\", ", dev, device_prop.name);
        if(dev == selected_device)
            printf("Selected, ");
        else
            printf("Unused, ");

        printf("CDNA %s\n", device_prop.gcnArchName);
        printf("CU%d%d, %zu [bytes]\n",
            device_prop.major, device_prop.minor,
            device_prop.totalGlobalMem);
    }
}

#if !defined(NO_MAIN_REPLACEMENT)

__host__ __device__
int fake_main(int, char**);

#ifndef LIBHIPCXX_GPU_COUNT
#define LIBHIPCXX_GPU_COUNT 1
#endif
static constexpr int HIP_GPU_COUNT = LIBHIPCXX_GPU_COUNT;

/********************************** GLOBALS ***********************************/
// All of these will have usable defaults, but can be redefined in tests
// for more interesting logic.

// Kernel launch parameters
int cuda_block_count = 1;
int cuda_thread_count = 1;
size_t cuda_shared_memory_size = 0;

// Global data to be accessed by any test (host or device side code)
__managed__ void* pRawGlobalData = nullptr;
// Index of currently running gpu. Determined by host for access on device.
__device__ int hip_gpu_index     = 0;
// Unique error code per device
__device__ int errorCodes[HIP_GPU_COUNT];
// Callback for host participation in tests
int (*hostSideWorkFunc)() = []() {return 0;};

__global__
void fake_main_kernel(int * ret)
{
  atomicCAS(ret, 0, fake_main(0, NULL));
}

// Unified main function - handles both standard and cooperative launch
int main(int argc, char** argv)
{
    // Common initialization
    cudaError_t err;
    HIP_CALL(err, cudaDeviceSynchronize());

    list_devices();

    int ret = fake_main(argc, argv);
    if (ret != 0)
    {
        return ret;
    }

    // Allocate one result slot per GPU so each kernel writes to its own int
    int * hip_ret = 0;
    HIP_CALL(err, hipMalloc(&hip_ret, sizeof(int[HIP_GPU_COUNT])));
    HIP_CALL(err, hipMemset(hip_ret, 0, sizeof(int[HIP_GPU_COUNT])));

    // Zero the per-device errorCodes symbol on each GPU before launches
    for (int gpuIndex = 0; gpuIndex < HIP_GPU_COUNT; ++gpuIndex)
    {
        HIP_CALL(err, hipSetDevice(gpuIndex));
        int zeroes[HIP_GPU_COUNT] = {0};
        HIP_CALL(err, hipMemcpyToSymbol(HIP_SYMBOL(errorCodes), zeroes, sizeof(zeroes)));
    }

    // Enable peer access between all GPUs
    for (int gpuIndex = 0; gpuIndex < HIP_GPU_COUNT; ++gpuIndex)
    {
        HIP_CALL(err, hipSetDevice(gpuIndex));
        for (int peerIndex = 0; peerIndex < HIP_GPU_COUNT; ++peerIndex)
        {
            if (peerIndex == gpuIndex)
                continue;
            HIP_CALL(err, hipDeviceEnablePeerAccess(peerIndex, 0));
        }
    }

    // Launch kernel on each GPU
    for (int gpuIndex = 0; gpuIndex < HIP_GPU_COUNT; ++gpuIndex)
    {
        HIP_CALL(err, hipSetDevice(gpuIndex));
        HIP_CALL(err, hipMemcpyToSymbol(HIP_SYMBOL(hip_gpu_index), &gpuIndex, sizeof(int)));

#if defined(USE_COOPERATIVE_LAUNCH)
        // ========== COOPERATIVE LAUNCH PATH ==========

        // Validate device supports cooperative launch
        cudaDeviceProp prop;
        HIP_CALL(err, cudaGetDeviceProperties(&prop, gpuIndex));

        if (prop.cooperativeLaunch == 0)
        {
            printf("[ERROR] Device %d does not support cooperative launch (cooperativeLaunch=0)\n", gpuIndex);
            printf("[ERROR] Device: %s\n", prop.name);
            printf("[ERROR] Test requires device with cooperativeLaunch=1\n");
            fflush(stdout);
            return -1;
        }

        // Check occupancy limits
        int maxBlocksPerSM = 0;
        err = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            fake_main_kernel,
            cuda_thread_count,
            cuda_shared_memory_size
        );

        if (err == hipSuccess)
        {
            int maxConcurrentBlocks = maxBlocksPerSM * prop.multiProcessorCount;
            if (cuda_block_count > maxConcurrentBlocks)
            {
                printf("[ERROR] Block count %d exceeds device limit %d\n",
                       cuda_block_count, maxConcurrentBlocks);
                printf("[ERROR] Device: %s (MPs: %d, max blocks/MP: %d)\n",
                       prop.name, prop.multiProcessorCount, maxBlocksPerSM);
                fflush(stdout);
                cudaFree(hip_ret);
                return -1;
            }
        }
        // Note: Silently ignore occupancy API failures - not critical

        // Prepare kernel arguments
        int * hip_ret_gpu = &hip_ret[gpuIndex];
        void* args[] = {&hip_ret_gpu};

        // Launch kernel cooperatively
        err = hipLaunchCooperativeKernel(
            (void*)fake_main_kernel,
            dim3(cuda_block_count),
            dim3(cuda_thread_count),
            args,
            cuda_shared_memory_size,
            nullptr  // Default stream
        );

        if (err != hipSuccess)
        {
            printf("[ERROR] hipLaunchCooperativeKernel failed: %d (%s)\n",
                   (int)err, cudaGetErrorString(err));
            printf("[ERROR] Launch parameters: blocks=%d, threads=%d, shared_mem=%lu\n",
                   cuda_block_count, cuda_thread_count, cuda_shared_memory_size);
            fflush(stdout);
            return -1;
        }

#else
        // ========== STANDARD LAUNCH PATH ==========

        fake_main_kernel<<<cuda_block_count, cuda_thread_count, cuda_shared_memory_size>>>(&hip_ret[gpuIndex]);

        HIP_CALL(err, cudaGetLastError());
#endif
    }

    auto hostWorkerErr = hostSideWorkFunc();
    if (hostWorkerErr != 0)
    {
      printf("[ERROR] CPU worker returned %d\n", hostWorkerErr);
      return hostWorkerErr;
    }

    // Synchronize all GPUs
    for (int gpuIndex = 0; gpuIndex < HIP_GPU_COUNT; ++gpuIndex)
    {
        HIP_CALL(err, hipSetDevice(gpuIndex));
        HIP_CALL(err, hipDeviceSynchronize());
    }

    // Retrieve all per-GPU results and return the first failure
    int host_ret[HIP_GPU_COUNT];
    HIP_CALL(err, hipMemcpy(host_ret, hip_ret, sizeof(int[HIP_GPU_COUNT]), hipMemcpyDeviceToHost));
    HIP_CALL(err, hipFree(hip_ret));

    for (int i = 0; i < HIP_GPU_COUNT; ++i)
    {
        if (host_ret[i] != 0)
        {
            printf("[ERROR] GPU %d kernel returned %d\n", i, host_ret[i]);
            return host_ret[i];
        }
    }
    return 0;
}

#if defined(__HIP_PLATFORM_AMD__)
#define main __device__ __host__ fake_main
#else
#define main fake_main
#endif

#endif // !defined(NO_MAIN_REPLACEMENT)
#endif // LIBCUDACXX_FORCE_INCLUDE_HIP
