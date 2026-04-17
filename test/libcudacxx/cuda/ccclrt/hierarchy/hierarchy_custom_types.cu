//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <iostream>

// NOTE(HIP/AMD): on HIP the upstream-named <cooperative_groups.h>
// does not exist; the HIP equivalent is
// <hip/hip_cooperative_groups.h>. Both populate
// "namespace cooperative_groups" so consumer code that uses cg::
// aliases works unchanged.
#if defined(__HIP_PLATFORM_AMD__)
#  include <hip/hip_cooperative_groups.h>
#else
#  include <cooperative_groups.h>
#endif // !__HIP_PLATFORM_AMD__
#include <host_device.cuh>

struct custom_level : public cuda::hierarchy_level_base<custom_level>
{
  using __product_type  = unsigned int;
  using __allowed_above = cuda::__allowed_levels<cuda::grid_level>;
  using __allowed_below = cuda::__allowed_levels<cuda::block_level>;
};

template <typename Level, typename Dims>
struct custom_level_dims : public cuda::hierarchy_level_desc<Level, Dims>
{
  int dummy;
  constexpr custom_level_dims()
      : cuda::hierarchy_level_desc<Level, Dims>() {};
};

struct custom_level_test
{
  template <typename DynDims>
  TEST_FUNC void operator()(const DynDims& dims) const
  {
    // todo: allow this after fixing CCCLRT_REQUIRE with clang-cuda
    // NOTE(HIP/AMD): clang-hip exhibits the same two-pass parsing
    // behaviour as clang-cuda -- the device pass parses the body of
    // host_device functions and tries to resolve the host-only
    // REQUIRE() symbols. Extend the upstream guard to skip the
    // device-pass parsing on HIP too.
#if !_CCCL_CUDA_COMPILER(CLANG) && !_CCCL_HIP_COMPILATION()
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, dims) == 84 * 1024);
    CCCLRT_REQUIRE(custom_level{}.count(cuda::grid, dims) == 42);
    CCCLRT_REQUIRE(cuda::gpu_thread.dims(cuda::grid, dims) == dim3(42 * 512, 2, 2));
    CCCLRT_REQUIRE(custom_level{}.dims(cuda::grid, dims) == dim3(42, 1, 1));
#endif // !_CCCL_CUDA_COMPILER(CLANG)
  }

  void run()
  {
    // Check extending hierarchy_level_desc with custom info
    custom_level_dims<cuda::block_level, cuda::std::extents<int, 64, 1, 1>> custom_block;
    custom_block.dummy     = 2;
    auto custom_dims       = cuda::make_hierarchy(cuda::grid_dims<256>(), cuda::cluster_dims<8>(), custom_block);
    auto custom_block_back = custom_dims.level(cuda::block);
    CCCLRT_REQUIRE(custom_block_back.dummy == 2);

    auto custom_dims_fragment = custom_dims.fragment(cuda::gpu_thread, cuda::block);
    auto custom_block_back2   = custom_dims_fragment.level(cuda::block);
    CCCLRT_REQUIRE(custom_block_back2.dummy == 2);

    // Check creating a custom level type works
    auto custom_level_dims = cuda::std::extents<cuda::dimensions_index_type, 2, 2, 2>();
    auto custom_hierarchy  = cuda::make_hierarchy(
      cuda::grid_dims(42),
      cuda::hierarchy_level_desc<custom_level, decltype(custom_level_dims)>(custom_level_dims),
      cuda::block_dims<256>());

    static_assert(cuda::gpu_thread.dims(custom_level(), custom_hierarchy) == dim3(512, 2, 2));
    static_assert(cuda::gpu_thread.count(custom_level(), custom_hierarchy) == 2048);

    test_host_dev(custom_hierarchy, *this);
  }
};

C2H_TEST("Custom level", "[hierarchy]")
{
  custom_level_test().run();
}
