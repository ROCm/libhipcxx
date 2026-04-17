//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef SUPPORT_HIERARCHY_QUERIES_H
#define SUPPORT_HIERARCHY_QUERIES_H

#include <cuda/hierarchy>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>

#include "test_macros.h"

template <class T, class Vec>
TEST_DEVICE_FUNC void test_result(cuda::hierarchy_query_result<T> res, Vec exp)
{
  assert(res.x == static_cast<T>(exp.x));
  assert(res.y == static_cast<T>(exp.y));
  assert(res.z == static_cast<T>(exp.z));
}

template <class IRes, class IExp, cuda::std::size_t... Exts>
TEST_DEVICE_FUNC void test_result(cuda::std::extents<IRes, Exts...> res, cuda::std::extents<IExp, Exts...> exp)
{
  for (cuda::std::size_t i = 0; i < sizeof...(Exts); ++i)
  {
    assert(res.extent(i) == static_cast<IRes>(exp.extent(i)));
  }
}

// <<<<<<< OLD CODE from 2ceb15d672 (5214850b75) - COMMENTED OUT
// // NOTE(HIP/AMD): templated on the vector type so the built-in dimension
// // variables (e.g. gridDim/blockDim, which are uint3 on CUDA but distinct
// // __hip_builtin_*_t types on HIP) can be passed directly. test_result only
// // uses .x/.y/.z, so this is portable across both backends.
// template <class Vec, class Level, class... Args>
// __device__ void test_dims(const Vec exp, const Level& level, Args... args)
// =======
template <class Level, class... Args>
TEST_DEVICE_FUNC void test_dims(const uint3 exp, const Level& level, Args... args)
// >>>>>>> END NEW CODE (5214850b75)
{
  test_result(level.dims(args...), exp);
  test_result(level.template dims_as<short>(args...), exp);
  test_result(level.template dims_as<int>(args...), exp);
  test_result(level.template dims_as<long long>(args...), exp);
  test_result(level.template dims_as<unsigned short>(args...), exp);
  test_result(level.template dims_as<unsigned int>(args...), exp);
  test_result(level.template dims_as<unsigned long long>(args...), exp);
}

template <class Level, class... Args>
TEST_DEVICE_FUNC void test_static_dims(const ulonglong3 exp, Level level, Args... args)
{
  static_assert(level.static_dims(args...).x != 0);
  test_result(level.static_dims(args...), exp);
}

template <class Exp, class Level, class... Args>
TEST_DEVICE_FUNC void test_extents(const Exp exp, const Level& level, Args... args)
{
  test_result(level.extents(args...), exp);
  test_result(level.template extents_as<short>(args...), exp);
  test_result(level.template extents_as<int>(args...), exp);
  test_result(level.template extents_as<long long>(args...), exp);
  test_result(level.template extents_as<unsigned short>(args...), exp);
  test_result(level.template extents_as<unsigned int>(args...), exp);
  test_result(level.template extents_as<unsigned long long>(args...), exp);
}

template <class Level, class... Args>
TEST_DEVICE_FUNC void test_static_count(Level level, Args... args)
{
  constexpr auto static_dims = level.static_dims(args...);
  if constexpr (static_dims.x != cuda::std::dynamic_extent && static_dims.y != cuda::std::dynamic_extent
                && static_dims.z != cuda::std::dynamic_extent)
  {
    static_assert(level.static_count(args...) == static_dims.x * static_dims.y * static_dims.z);
  }
  else
  {
    static_assert(level.static_count(args...) == cuda::std::dynamic_extent);
  }
}

template <class Level, class... Args>
TEST_DEVICE_FUNC void test_count(const cuda::std::size_t exp, const Level& level, Args... args)
{
  assert(level.count(args...) == exp);
  assert(level.template count_as<short>(args...) == static_cast<short>(exp));
  assert(level.template count_as<int>(args...) == static_cast<int>(exp));
  assert(level.template count_as<long long>(args...) == static_cast<long long>(exp));
  assert(level.template count_as<unsigned short>(args...) == static_cast<unsigned short>(exp));
  assert(level.template count_as<unsigned int>(args...) == static_cast<unsigned int>(exp));
  assert(level.template count_as<unsigned long long>(args...) == static_cast<unsigned long long>(exp));
}

// <<<<<<< OLD CODE from 2ceb15d672 (5214850b75) - COMMENTED OUT
// // NOTE(HIP/AMD): templated first parameter; see test_dims above.
// template <class Vec, class Level, class... Args>
// __device__ void test_index(const Vec exp, const Level& level, Args... args)
// =======
template <class Level, class... Args>
TEST_DEVICE_FUNC void test_index(const uint3 exp, const Level& level, Args... args)
// >>>>>>> END NEW CODE (5214850b75)
{
  test_result(level.index(args...), exp);
  test_result(level.template index_as<short>(args...), exp);
  test_result(level.template index_as<int>(args...), exp);
  test_result(level.template index_as<long long>(args...), exp);
  test_result(level.template index_as<unsigned short>(args...), exp);
  test_result(level.template index_as<unsigned int>(args...), exp);
  test_result(level.template index_as<unsigned long long>(args...), exp);
}

template <class Level, class... Args>
TEST_DEVICE_FUNC void test_rank(const cuda::std::size_t exp, const Level& level, Args... args)
{
  assert(level.rank(args...) == exp);
  assert(level.template rank_as<short>(args...) == static_cast<short>(exp));
  assert(level.template rank_as<int>(args...) == static_cast<int>(exp));
  assert(level.template rank_as<long long>(args...) == static_cast<long long>(exp));
  assert(level.template rank_as<unsigned short>(args...) == static_cast<unsigned short>(exp));
  assert(level.template rank_as<unsigned int>(args...) == static_cast<unsigned int>(exp));
  assert(level.template rank_as<unsigned long long>(args...) == static_cast<unsigned long long>(exp));
}

template <class... Args>
TEST_DEVICE_FUNC constexpr cuda::std::size_t mul_static_extents(Args... args)
{
  if (((args == cuda::std::dynamic_extent) || ...))
  {
    return cuda::std::dynamic_extent;
  }
  else
  {
    return (cuda::std::size_t{1} * ... * args);
  }
}

#endif // SUPPORT_HIERARCHY_QUERIES_H
