//===----------------------------------------------------------------------===//
//
// Part of libhip++ (derived from libcu++),
// the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
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


// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: gcc-4
// TODO(HIP): Add support for functional header.
// UNSUPPORTED: hipcc

#include <hip/functional>

#include <hip/std/cassert>

template <class T, class Fn, class... As>
__host__ __device__
void test_proclaim_return_type(Fn&& fn, T expected, As... as)
{
  {
    auto f = hip::proclaim_return_type<T>(hip::std::forward<Fn>(fn));

    assert(f(as...) == expected);
    assert(hip::std::move(f)(as...) == expected);
  }

  {
    const auto f = hip::proclaim_return_type<T>(fn);

    assert(f(as...) == expected);
    assert(hip::std::move(f)(as...) == expected);
  }
}

struct hd_callable
{
  __host__ __device__ int operator()() const& { return 42; }
  __host__ __device__ int operator()() const&& { return 42; }
};

#if !defined(__CUDACC_RTC__)
struct h_callable
{
  __host__ int operator()() const& { return 42; }
  __host__ int operator()() const&& { return 42; }
};
#endif

struct d_callable
{
  __device__ int operator()() const& { return 42; }
  __device__ int operator()() const&& { return 42; }
};

int main(int argc, char ** argv)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define TEST_SPECIFIER(...)                                                    \
  {                                                                            \
    test_proclaim_return_type<double>([] __VA_ARGS__ () { return 42.0; },      \
                                      42.0);                                   \
    test_proclaim_return_type<int>([] __VA_ARGS__ (int v) { return v * 2; },   \
                                   42, 21);                                    \
                                                                               \
    int v = 42;                                                                \
    int* vp = &v;                                                              \
    test_proclaim_return_type<int&>(                                           \
        [vp] __VA_ARGS__ () -> int& { return *vp; }, v);                       \
  }

#if !defined(__CUDACC_RTC__)
  TEST_SPECIFIER(__device__)
  TEST_SPECIFIER(__host__ __device__)
#endif
  TEST_SPECIFIER()
#undef TEST_SPECIFIER

  test_proclaim_return_type<int>(hd_callable{}, 42);
  test_proclaim_return_type<int>(d_callable{}, 42);
#else
  test_proclaim_return_type<int>(h_callable{}, 42);
#endif

  return 0;
}
