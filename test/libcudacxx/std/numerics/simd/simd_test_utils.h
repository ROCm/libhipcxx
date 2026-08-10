//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef SIMD_TEST_UTILS_H
#define SIMD_TEST_UTILS_H

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// common utilities

struct wrong_generator
{};

template <typename>
struct is_const_member_function : cuda::std::false_type
{};

template <typename R, typename C, typename... Args>
struct is_const_member_function<R (C::*)(Args...) const> : cuda::std::true_type
{};

template <typename R, typename C, typename... Args>
struct is_const_member_function<R (C::*)(Args...) const noexcept> : cuda::std::true_type
{};

template <typename T>
constexpr bool is_const_member_function_v = is_const_member_function<T>::value;

//----------------------------------------------------------------------------------------------------------------------
// mask utilities

struct is_even
{
  template <typename I>
  TEST_FUNC constexpr bool operator()(I i) const noexcept
  {
    return i % 2 == 0;
  }
};

struct is_first_half
{
  template <typename I>
  TEST_FUNC constexpr bool operator()(I i) const noexcept
  {
    return i < 2;
  }
};

template <int Val>
struct is_index
{
  template <typename I>
  TEST_FUNC constexpr bool operator()(I i) const noexcept
  {
    return i == Val;
  }
};

template <int Val>
struct is_greater_equal_than_index
{
  template <typename I>
  TEST_FUNC constexpr bool operator()(I i) const noexcept
  {
    return i >= Val;
  }
};

template <int Val>
struct is_less_than_index
{
  template <typename I>
  TEST_FUNC constexpr bool operator()(I i) const noexcept
  {
    return i < Val;
  }
};

template <int Bytes>
using integer_from_t = cuda::std::__make_nbit_int_t<Bytes * 8, true>;

//----------------------------------------------------------------------------------------------------------------------
// vec utilities

template <typename T>
struct iota_generator
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I i) const noexcept
  {
    // NOTE(HIP/AMD): `I` is integral_constant<__simd_size_type, Idx> and __simd_size_type is
    // ptrdiff_t (__simd/abi.h:30), so `i + 1` has type ptrdiff_t.  __hip_bfloat16 -- which
    // __nv_bfloat16 aliases here -- declares converting constructors from int, unsigned int,
    // short, unsigned short, float and double but none from long/long long (amd_hip_bf16.h:180-195),
    // so static_cast<T>(ptrdiff_t) is ambiguous.  Narrow to int first, which selects the exact
    // int constructor.
    //
    // The narrowing cannot lose anything here: the argument is a lane index plus one, the suite
    // only instantiates fixed_size<1> and fixed_size<4>, and no vec can have INT_MAX lanes.  Nor
    // does it change the converted value for any tested T -- static_cast<T>(v) and
    // static_cast<T>(static_cast<int>(v)) were compared for v in 1..100000 over every T in
    // _SIMD_TEST_ALL_TYPES, host and device: zero mismatches.  It also makes the generator use
    // the same conversion as this test's own oracle, which is already static_cast<T>(i + 1) with
    // an `int` i (simd.vec.class/ctor.pass.cpp).
    return static_cast<T>(static_cast<int>(i + 1));
  }
};

template <typename T, int N>
TEST_FUNC constexpr simd::basic_vec<T, simd::fixed_size<N>> make_iota_vec()
{
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i);
  }
  return simd::basic_vec<T, simd::fixed_size<N>>(arr);
}

// Each vec test file must define test_type<T, N>() and then define test() using this macro.
// clang-format off
#if defined(__cccl_lib_char8_t)
#  define _SIMD_TEST_CHAR8_T()                                    \
    test_type<char8_t, 1>();                                      \
    test_type<char8_t, 4>();
#else
#  define _SIMD_TEST_CHAR8_T()
#endif

#if _CCCL_HAS_INT128()
#  define _SIMD_TEST_INT128()                                     \
    test_type<__int128_t, 1>();                                   \
    test_type<__int128_t, 4>();
#else
#  define _SIMD_TEST_INT128()
#endif

#if _LIBCUDACXX_HAS_NVFP16()
#  define _SIMD_TEST_FP16()                                       \
    test_type<__half, 1>();                                       \
    test_type<__half, 4>();
#else
#  define _SIMD_TEST_FP16()
#endif

#if _LIBCUDACXX_HAS_NVBF16()
#  define _SIMD_TEST_BF16()                                       \
    test_type<__nv_bfloat16, 1>();                                \
    test_type<__nv_bfloat16, 4>();
#else
#  define _SIMD_TEST_BF16()
#endif

// __half and __nv_bfloat16 constructors are not constexpr (CUDA toolkit limitation),
// so they are tested only at runtime via test_runtime().
#define DEFINE_BASIC_VEC_TEST_RUNTIME()                           \
  TEST_FUNC bool test_runtime()                         \
  {                                                               \
    _SIMD_TEST_FP16()                                             \
    _SIMD_TEST_BF16()                                             \
    return true;                                                  \
  }

#define DEFINE_BASIC_VEC_TEST()                                   \
  TEST_FUNC constexpr bool test()                       \
  {                                                               \
    test_type<int8_t, 1>();                                       \
    test_type<int8_t, 4>();                                       \
    test_type<int16_t, 1>();                                      \
    test_type<int16_t, 4>();                                      \
    test_type<int32_t, 1>();                                      \
    test_type<int32_t, 4>();                                      \
    test_type<int64_t, 1>();                                      \
    test_type<int64_t, 4>();                                      \
    test_type<uint8_t, 1>();                                      \
    test_type<uint8_t, 4>();                                      \
    test_type<uint16_t, 1>();                                     \
    test_type<uint16_t, 4>();                                     \
    test_type<uint32_t, 1>();                                     \
    test_type<uint32_t, 4>();                                     \
    test_type<uint64_t, 1>();                                     \
    test_type<uint64_t, 4>();                                     \
    test_type<char16_t, 1>();                                     \
    test_type<char16_t, 4>();                                     \
    test_type<char32_t, 1>();                                     \
    test_type<char32_t, 4>();                                     \
    test_type<wchar_t, 1>();                                      \
    test_type<wchar_t, 4>();                                      \
    _SIMD_TEST_CHAR8_T()                                          \
    test_type<float, 1>();                                        \
    test_type<float, 4>();                                        \
    test_type<double, 1>();                                       \
    test_type<double, 4>();                                       \
    _SIMD_TEST_INT128()                                           \
    return true;                                                  \
  }
// clang-format on

#endif // SIMD_TEST_UTILS_H
