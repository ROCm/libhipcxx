//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// <cuda/complex>

#include <cuda/__complex_>
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

// <<<<<<< OLD CODE from b5d4ca3cf8 (e5037ea8b4) - COMMENTED OUT
// #if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
// #  include <complex>
// #endif // !_CCCL_COMPILER(NVRTC) && !_CCCL_COMPILER_HIPRTC
// =======
#if _CCCL_HOSTED()
#  include <complex>
#endif // _CCCL_HOSTED()
// >>>>>>> END NEW CODE (e5037ea8b4)

#include "test_macros.h"

template <class T>
TEST_FUNC void test_deduction()
{
  // 1. Test cuda::complex(T)
  {
    [[maybe_unused]] cuda::complex c{T{}};
    static_assert(cuda::std::is_same_v<T, typename decltype(c)::value_type>);
  }

  // 2. Test cuda::complex(T, T)
  {
    [[maybe_unused]] cuda::complex c{T{}, T{}};
    static_assert(cuda::std::is_same_v<T, typename decltype(c)::value_type>);
  }

  // 3. Test cuda::complex(cuda::complex)
  {
    [[maybe_unused]] cuda::complex c{cuda::complex<T>{}};
    static_assert(cuda::std::is_same_v<T, typename decltype(c)::value_type>);
  }

  // 4. Test cuda::complex(cuda::std::complex)
  {
    [[maybe_unused]] cuda::complex c{cuda::std::complex<T>{}};
    static_assert(cuda::std::is_same_v<T, typename decltype(c)::value_type>);
  }

  // 5. Test cuda::complex(std::complex)
// <<<<<<< OLD CODE from b5d4ca3cf8 (e5037ea8b4) - COMMENTED OUT
// #if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
// =======
#if _CCCL_HOSTED()
// >>>>>>> END NEW CODE (e5037ea8b4)
  // std::complex is not required to support other than standard floating-point types
  if constexpr (cuda::std::__is_std_fp_v<T>)
  {
    NV_IF_TARGET(NV_IS_HOST,
                 ([[maybe_unused]] cuda::complex c{std::complex<T>{}}; //
                  assert((cuda::std::is_same_v<T, typename decltype(c)::value_type>) );))
  }
// <<<<<<< OLD CODE from b5d4ca3cf8 (e5037ea8b4) - COMMENTED OUT
// #endif // !_CCCL_COMPILER(NVRTC) && !_CCCL_COMPILER_HIPRTC
// =======
#endif // _CCCL_HOSTED()
// >>>>>>> END NEW CODE (e5037ea8b4)

  // 6. Test cuda::complex(tuple-like)
#if _CCCL_STD_VER >= 2020
  {
    T value{};

    [[maybe_unused]] cuda::complex c1{cuda::std::tuple<T, T>{value, value}};
    static_assert(cuda::std::is_same_v<T, typename decltype(c1)::value_type>);

    [[maybe_unused]] cuda::complex c2{cuda::std::tuple<const T, T&>{value, value}};
    static_assert(cuda::std::is_same_v<T, typename decltype(c2)::value_type>);

    [[maybe_unused]] cuda::complex c3{cuda::std::tuple<const T&, T&&>{value, T{value}}};
    static_assert(cuda::std::is_same_v<T, typename decltype(c3)::value_type>);
  }
#endif // _CCCL_STD_VER >= 2020
}

TEST_FUNC void test()
{
  test_deduction<float>();
  test_deduction<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_deduction<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_deduction<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_deduction<signed char>();
  test_deduction<signed short>();
  test_deduction<signed int>();
  test_deduction<signed long>();
  test_deduction<signed long long>();
#if _CCCL_HAS_INT128()
  test_deduction<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_deduction<unsigned char>();
  test_deduction<unsigned short>();
  test_deduction<unsigned int>();
  test_deduction<unsigned long>();
  test_deduction<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_deduction<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

int main(int, char**)
{
  test();
  return 0;
}
