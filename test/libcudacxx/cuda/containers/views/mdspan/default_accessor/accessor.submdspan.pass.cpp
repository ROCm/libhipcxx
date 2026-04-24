//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <<<<<<< OLD CODE from e46ae1ed13 (0a2929ae18) - COMMENTED OUT
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
// =======
// XFAIL: enable-tile
// error: a non-__tile__ variable cannot be used in tile code
// >>>>>>> END NEW CODE (0a2929ae18)

#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY
#include <cuda/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename Mdspan>
TEST_FUNC void test_submdspan(int* ptr)
{
  Mdspan md{ptr, cuda::std::dims<1>{4}};
  auto submd = cuda::std::submdspan(md, cuda::std::pair{1, 3});
  if constexpr (cuda::is_device_accessible_v<Mdspan>)
  {
    NV_IF_TARGET(NV_IS_DEVICE, (assert(submd(0) == 2); assert(submd(1) == 3);))
  }
  if constexpr (cuda::is_host_accessible_v<Mdspan>)
  {
    NV_IF_TARGET(NV_IS_HOST, (assert(submd(0) == 2); assert(submd(1) == 3);))
  }
  unused(submd);
}

// <<<<<<< OLD CODE from 2ceb15d672 (5214850b75) - COMMENTED OUT
// // NOTE(HIP/AMD): __managed__ globals are not detected as managed by HIP's
// // hipPointerGetAttributes (see WAR-18 in CHANGELOG_v3.1.md). The strict
// // HIP check in cuda::__managed_accessor only accepts hipMallocManaged()
// // allocations, so __managed__ globals can no longer be used with
// // cuda::managed_mdspan on HIP. Skip the managed_mdspan path on HIP.
// #if !defined(__HIP_PLATFORM_AMD__)
// __device__ __managed__ int managed_array[] = {1, 2, 3, 4};
// #endif // !__HIP_PLATFORM_AMD__
// =======
_CCCL_DEVICE __managed__ int managed_array[] = {1, 2, 3, 4};
// >>>>>>> END NEW CODE (5214850b75)

TEST_FUNC void test_submdspan()
{
  int array[] = {1, 2, 3, 4};
  test_submdspan<cuda::host_mdspan<int, cuda::std::dims<1>>>(array);
  test_submdspan<cuda::device_mdspan<int, cuda::std::dims<1>>>(array);
#if !defined(__HIP_PLATFORM_AMD__)
  test_submdspan<cuda::managed_mdspan<int, cuda::std::dims<1>>>(managed_array);
#endif // !__HIP_PLATFORM_AMD__
}

int main(int, char**)
{
  test_submdspan();
  return 0;
}
