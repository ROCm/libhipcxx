//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <<<<<<< OLD CODE from a8b8e0984a (98ec5e3d4f) - COMMENTED OUT
// // MIT License
// //
// // Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// //
// // Permission is hereby granted, free of charge, to any person obtaining a copy
// // of this software and associated documentation files (the "Software"), to deal
// // in the Software without restriction, including without limitation the rights
// // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// // copies of the Software, and to permit persons to whom the Software is
// // furnished to do so, subject to the following conditions:
// //
// // The above copyright notice and this permission notice shall be included in all
// // copies or substantial portions of the Software.
// //
// // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// // SOFTWARE.
//
// // todo: enable with nvrtc
// // UNSUPPORTED: nvrtc, hiprtc
//
// =======
// >>>>>>> END NEW CODE (98ec5e3d4f)
#include <cuda/hierarchy>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "hierarchy_queries.h"
#include "test_macros.h"

TEST_DEVICE_FUNC void test_cluster()
{
  constexpr cuda::std::size_t dext = cuda::std::dynamic_extent;

  uint3 dims{gridDim.x, gridDim.y, gridDim.z};
  NV_IF_TARGET(NV_PROVIDES_SM_90, (dims = __clusterGridDimInClusters();))

  uint3 index{blockIdx.x, blockIdx.y, blockIdx.z};
  NV_IF_TARGET(NV_PROVIDES_SM_90, (index = __clusterIdx();))

  // 1. Test cuda::cluster.dims(x)
  test_dims(dims, cuda::cluster, cuda::grid);

  // 2. Test cuda::cluster.static_dims(x)
  test_static_dims(ulonglong3{dext, dext, dext}, cuda::cluster, cuda::grid);

  // 3. Test cuda::cluster.extents(x)
  test_extents(cuda::std::dims<3, unsigned>{dims.x, dims.y, dims.z}, cuda::cluster, cuda::grid);

  // 4. Test cuda::cluster.static_count(x)
  test_static_count(cuda::cluster, cuda::grid);

  // 5. Test cuda::cluster.count(x)
  test_count(cuda::std::size_t{dims.z} * dims.y * dims.x, cuda::cluster, cuda::grid);

  // 6. test cuda::cluster.index(x)
  test_index(index, cuda::cluster, cuda::grid);

  // 7. Test cuda::cluster.rank(x)
  {
    const cuda::std::size_t exp = (index.z * dims.y + index.y) * dims.x + index.x;
    test_rank(exp, cuda::cluster, cuda::grid);
  }
}

#if !_CCCL_COMPILER(NVRTC)
__global__ void test_kernel()
{
  test_cluster();
}

void test()
{
  [[maybe_unused]] int cc_major{};
  assert(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess);

  // thread block clusters require compute capability at least 9.0
  [[maybe_unused]] const bool enable_clusters = cc_major >= 9;

  test_kernel<<<1, 128>>>();
  test_kernel<<<128, 1>>>();
  test_kernel<<<dim3{2, 3}, dim3{4, 2}>>>();
  test_kernel<<<dim3{2, 3, 4}, dim3{4, 2, 8}>>>();
#if !defined(__HIP_PLATFORM_AMD__) // NOTE(HIP/AMD): thread-block clusters are NVIDIA-only
  if (enable_clusters)
  {
    cudaLaunchAttribute attribute[1]{};
    attribute[0].id               = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 4;
    attribute[0].val.clusterDim.y = 2;
    attribute[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim  = {12, 10, 3};
    config.blockDim = {2, 8, 4};
    config.attrs    = attribute;
    config.numAttrs = 1;

    void* pargs[1]{};
    assert(cudaLaunchKernelExC(&config, (const void*) test_kernel, pargs) == cudaSuccess);
  }
#endif // !defined(__HIP_PLATFORM_AMD__)

  assert(cudaDeviceSynchronize() == cudaSuccess);
}
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test();), (test_cluster();))
  return 0;
}
