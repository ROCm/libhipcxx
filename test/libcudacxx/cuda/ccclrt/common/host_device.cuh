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

#ifndef __COMMON_HOST_DEVICE_H__
#define __COMMON_HOST_DEVICE_H__

#include <cuda/hierarchy>

#include "utility.cuh"

template <typename Dims, typename Lambda>
void __global__ lambda_launcher(const Dims dims, const Lambda lambda)
{
  lambda(dims);
}

template <typename Comparator, unsigned int FilterArch>
bool arch_filter(const cudaDeviceProp& props)
{
  int act_arch = props.major * 10 + props.minor;
  if (Comparator()(act_arch, FilterArch))
  {
    return true;
  }
  return false;
}

static bool skip_host_exec(bool (* /* filter */)(const cudaDeviceProp&))
{
  return false;
}

static bool skip_device_exec(bool (*filter)(const cudaDeviceProp&))
{
  cudaDeviceProp props;
  CUDART(cudaGetDeviceProperties(&props, 0));
  return filter(props);
}

template <typename Dims, typename Lambda, typename... Filters>
void test_host_dev(const Dims& dims, const Lambda& lambda, const Filters&... filters)
{
  SECTION("Host execution")
  {
    if ((... && !skip_host_exec(filters)))
    {
      // host testing
      lambda(dims);
    }
  }

  SECTION("Device execution")
  {
    // Asymmetrical but cleaner
    if ((... || skip_device_exec(filters)))
    {
      return;
    }

    cudaLaunchConfig_t config = {};
    config.gridDim            = {0};
    cudaLaunchAttribute attrs[1];
    config.attrs = &attrs[0];

    config.blockDim = dims.extents(cuda::gpu_thread, cuda::block);
    config.gridDim  = dims.extents(cuda::block, cuda::grid);

    if constexpr (cuda::has_level_v<cuda::cluster_level, decltype(dims)>)
    {
#if defined(__HIP_PLATFORM_AMD__)
      // NOTE(HIP/AMD): cluster dimension is a Hopper (sm_90+) NVPTX
      // feature; AMDGCN has no equivalent. The HIP runtime does not
      // expose 'cudaLaunchAttributeClusterDimension' nor a
      // 'clusterDim' member on hipLaunchAttributeValue. The only
      // place this branch fires is when the user constructed a
      // hierarchy with cuda::cluster_level, which only makes sense
      // on Hopper -- on HIP this branch is dead code.
      config.numAttrs = 0;
#else
      dim3 cluster_dims                            = dims.extents(cuda::block, cuda::cluster);
      config.attrs[config.numAttrs].id             = cudaLaunchAttributeClusterDimension;
      config.attrs[config.numAttrs].val.clusterDim = {cluster_dims.x, cluster_dims.y, cluster_dims.z};
      config.numAttrs                              = 1;
#endif
    }
    else
    {
      config.numAttrs = 0;
    }

    // device testing
    CUDART(cudaLaunchKernelEx(&config, lambda_launcher<Dims, Lambda>, dims, lambda));
    CUDART(cudaDeviceSynchronize());
  }
}

template <typename Fn, typename Tuple>
void apply_each(const Fn& fn, const Tuple& tuple)
{
  cuda::std::apply(
    [&](const auto&... elems) {
      (fn(elems), ...);
    },
    tuple);
}

#endif // __COMMON_HOST_DEVICE_H__
