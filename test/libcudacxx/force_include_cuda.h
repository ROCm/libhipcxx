//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// We use <stdio.h> instead of <iostream> to avoid relying on the host system's
// C++ standard library.
#ifndef _LIBCUDACXX_FORCE_INCLUDE_H
#define _LIBCUDACXX_FORCE_INCLUDE_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void list_devices()
{
  int device_count;
  hipGetDeviceCount(&device_count);
  printf("CUDA devices found: %d\n", device_count);

  int selected_device;
  hipGetDevice(&selected_device);

  for (int dev = 0; dev < device_count; ++dev)
  {
    hipDeviceProp_t device_prop;
    hipGetDeviceProperties(&device_prop, dev);

    printf("Device %d: \"%s\", ", dev, device_prop.name);
    if (dev == selected_device)
    {
      printf("Selected, ");
    }
    else
    {
      printf("Unused, ");
    }

    printf("SM%d%d, %zu [bytes]\n", device_prop.major, device_prop.minor, device_prop.totalGlobalMem);
  }
}

#ifdef __CUDACC_TILE__
__tile__
#endif // __CUDACC_TILE__
  __host__ __device__ int
  fake_main(int, char**);

int cuda_thread_count = 1;
int cuda_cluster_size = 1;

#ifdef __CUDACC_TILE__
__tile_global__
#else // ^^^ __CUDACC_TILE__ ^^^ / vvv !__CUDACC_TILE__ vvv
__global__
#endif // !__CUDACC_TILE__
  void
  fake_main_kernel(int* ret)
{
  // We need to use atomicCAS to make sure we don't overwrite previous non-zero value.
  atomicCAS(ret, 0, fake_main(0, nullptr));
}

#define CUDA_CALL(err, ...)                                                                              \
  do                                                                                                     \
  {                                                                                                      \
    err = __VA_ARGS__;                                                                                   \
    if (err != hipSuccess)                                                                              \
    {                                                                                                    \
      printf("CUDA ERROR, line %d: %s: %s\n", __LINE__, hipGetErrorName(err), hipGetErrorString(err)); \
      exit(1);                                                                                           \
    }                                                                                                    \
  } while (false)

int main(int argc, char** argv)
{
  // Check if the CUDA driver/runtime are installed and working for sanity.
  hipError_t err;
  CUDA_CALL(err, hipDeviceSynchronize());

  printf("Testing on host:\n");
  fflush(stdout);
  int ret = fake_main(argc, argv);
  if (ret != 0)
  {
    printf("Host testing returned failure\n");
    return ret;
  }

  printf("Testing on device:\n");
  fflush(stdout);
  list_devices();
  int* cuda_ret = 0;
  CUDA_CALL(err, hipMalloc(&cuda_ret, sizeof(int)));

  if (cuda_cluster_size > 1)
  {
    hipLaunchAttribute attribute[1];
    attribute[0].id               = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cuda_cluster_size; // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    hipLaunchConfig_t config = {
      dim3(cuda_cluster_size), // grid dim
      dim3(cuda_thread_count), // block dim
      0, // dynamic smem bytes
      0, // stream
      attribute, // attributes
      1, // number of attributes
    };
    CUDA_CALL(err, hipLaunchKernelEx(&config, fake_main_kernel, cuda_ret));
  }
  else
  {
    fake_main_kernel<<<1, cuda_thread_count>>>(cuda_ret);
  }

  CUDA_CALL(err, hipGetLastError());
  CUDA_CALL(err, hipDeviceSynchronize());
  CUDA_CALL(err, hipMemcpy(&ret, cuda_ret, sizeof(int), hipMemcpyDeviceToHost));
  CUDA_CALL(err, hipFree(cuda_ret));
  fflush(stdout);

  if (ret != 0)
  {
    printf("Device testing returned failure\n");
  }
  return ret;
}

#ifdef __CUDACC_TILE__
#  define main(...) __tile__ __host__ __device__ fake_main(__VA_ARGS__)
#else // ^^^ __CUDACC_TILE__ ^^^ / vvv !__CUDACC_TILE__ vvv
#  define main(...) __host__ __device__ fake_main(__VA_ARGS__)
#endif // !__CUDACC_TILE__

#endif
