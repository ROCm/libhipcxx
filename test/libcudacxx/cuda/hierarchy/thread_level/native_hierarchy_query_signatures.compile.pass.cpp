//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <<<<<<< OLD CODE from 62028ff6bf (5b85fe8f16) - COMMENTED OUT
// // <<<<<<< OLD CODE from a8b8e0984a (98ec5e3d4f) - COMMENTED OUT
// // // MIT License
// // //
// // // Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// // //
// // // Permission is hereby granted, free of charge, to any person obtaining a copy
// // // of this software and associated documentation files (the "Software"), to deal
// // // in the Software without restriction, including without limitation the rights
// // // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// // // copies of the Software, and to permit persons to whom the Software is
// // // furnished to do so, subject to the following conditions:
// // //
// // // The above copyright notice and this permission notice shall be included in all
// // // copies or substantial portions of the Software.
// // //
// // // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// // // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// // // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// // // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// // // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// // // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// // // SOFTWARE.
// //
// // // todo: enable with nvrtc
// // // UNSUPPORTED: nvrtc
// //
// // =======
// // >>>>>>> END NEW CODE (98ec5e3d4f)
// =======
// UNSUPPORTED: enable-tile
// error: accessing gridDim/blockDim/blockIdx/threadIdx/warpSize is unsupported in tile code

// >>>>>>> END NEW CODE (5b85fe8f16)
#include <cuda/hierarchy>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

// <<<<<<< OLD CODE from 2ceb15d672 (5214850b75) - COMMENTED OUT
// // NOTE(HIP/AMD): the warp/wavefront size is 32 on NVIDIA but wave32/wave64 on
// // AMD, so the static warp-level extent below must use the wave size.
// #if _CCCL_HIP_COMPILATION()
// #  define TEST_WARP_SIZE _CCCL_HIP_WAVE_SIZE
// #else
// #  define TEST_WARP_SIZE 32
// #endif
// =======
#include "test_macros.h"
// >>>>>>> END NEW CODE (5214850b75)

template <class Level>
TEST_DEVICE_FUNC void test_query_signatures(const Level& level)
{
  // 1. Test cuda::thread_level::dims(x) signature.
  static_assert(
    cuda::std::is_same_v<cuda::hierarchy_query_result<unsigned>, decltype(cuda::thread_level::dims(level))>);
  static_assert(noexcept(cuda::thread_level::dims(level)));

  // 2. Test cuda::thread_level::static_dims(x) signature.
  static_assert(cuda::std::is_same_v<cuda::hierarchy_query_result<cuda::std::size_t>,
                                     decltype(cuda::thread_level::static_dims(level))>);
  static_assert(noexcept(cuda::thread_level::static_dims(level)));

  // 3. Test cuda::thread_level::extents(x) signature.
  using ExtentsRet = cuda::std::conditional_t<cuda::std::is_same_v<Level, cuda::warp_level>,
                                              cuda::std::extents<unsigned, TEST_WARP_SIZE>,
                                              cuda::std::dims<3, unsigned>>;
  static_assert(cuda::std::is_same_v<ExtentsRet, decltype(cuda::thread_level::extents(level))>);
  static_assert(noexcept(cuda::thread_level::extents(level)));

  // 4. Test cuda::thread_level::static_count(x) signature.
  static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::thread_level::static_count(level))>);
  static_assert(noexcept(cuda::thread_level::static_count(level)));

  // 5. Test cuda::thread_level::count(x) signature.
  static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::thread_level::count(level))>);
  static_assert(noexcept(cuda::thread_level::count(level)));

  // 6. Test cuda::thread_level::index(x) signature.
  static_assert(
    cuda::std::is_same_v<cuda::hierarchy_query_result<unsigned>, decltype(cuda::thread_level::index(level))>);
  static_assert(noexcept(cuda::thread_level::index(level)));

  // 7. Test cuda::thread_level::rank(x) signature.
  static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::thread_level::rank(level))>);
  static_assert(noexcept(cuda::thread_level::rank(level)));
}

template <class T, class Level>
TEST_DEVICE_FUNC void test_query_as_signatures(const Level& level)
{
  // 1. Test cuda::thread_level::dims(x) signature.
  static_assert(cuda::std::is_same_v<cuda::hierarchy_query_result<T>, decltype(cuda::thread_level::dims_as<T>(level))>);
  static_assert(noexcept(cuda::thread_level::dims_as<T>(level)));

  // 2. Test cuda::thread_level::extents(x) signature.
  using ExtentsRet = cuda::std::
    conditional_t<cuda::std::is_same_v<Level, cuda::warp_level>, cuda::std::extents<T, TEST_WARP_SIZE>, cuda::std::dims<3, T>>;
  static_assert(cuda::std::is_same_v<ExtentsRet, decltype(cuda::thread_level::extents_as<T>(level))>);
  static_assert(noexcept(cuda::thread_level::extents_as<T>(level)));

  // 3. Test cuda::thread_level::count(x) signature.
  static_assert(cuda::std::is_same_v<T, decltype(cuda::thread_level::count_as<T>(level))>);
  static_assert(noexcept(cuda::thread_level::count_as<T>(level)));

  // 4. Test cuda::thread_level::index(x) signature.
  static_assert(
    cuda::std::is_same_v<cuda::hierarchy_query_result<T>, decltype(cuda::thread_level::index_as<T>(level))>);
  static_assert(noexcept(cuda::thread_level::index_as<T>(level)));

  // 5. Test cuda::thread_level::rank(x) signature.
  static_assert(cuda::std::is_same_v<T, decltype(cuda::thread_level::rank_as<T>(level))>);
  static_assert(noexcept(cuda::thread_level::rank_as<T>(level)));
}

template <class InLevel>
TEST_DEVICE_FUNC void test(const InLevel& in_level)
{
  test_query_signatures(in_level);
  test_query_as_signatures<short>(in_level);
  test_query_as_signatures<int>(in_level);
  test_query_as_signatures<long long>(in_level);
  test_query_as_signatures<unsigned short>(in_level);
  test_query_as_signatures<unsigned int>(in_level);
  test_query_as_signatures<unsigned long long>(in_level);
}

TEST_DEVICE_FUNC void test()
{
  test(cuda::warp);
  test(cuda::block);
  test(cuda::cluster);
  test(cuda::grid);
}

int main(int, char**)
{
  return 0;
}
