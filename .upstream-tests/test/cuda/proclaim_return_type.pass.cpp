//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

#include <cuda/functional>

#include <cuda/std/cassert>

#include "test_macros.h"

template <class T, class Fn, class... As>
__host__ __device__
void test_proclaim_return_type(Fn&& fn, T expected, As... as)
{
  {
    auto f1 = cuda::proclaim_return_type<T>(cuda::std::forward<Fn>(fn));

    assert(f1(as...) == expected);

    auto f2{f1};
    assert(cuda::std::move(f2)(as...) == expected);
  }

  {
    const auto f1 = cuda::proclaim_return_type<T>(fn);

    assert(f1(as...) == expected);

    auto f2{f1};
    assert(cuda::std::move(f2)(as...) == expected);
  }
}

struct hd_callable
{
  __host__ __device__ int operator()() const& { return 42; }
  __host__ __device__ int operator()() const&& { return 42; }
};

#if !defined(TEST_COMPILER_NVRTC)
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
  int v = 42;
  int* vp = &v;

  test_proclaim_return_type<int>(hd_callable{}, 42);
  test_proclaim_return_type<double>([]   { return 42.0; }, 42.0);
  test_proclaim_return_type<int>   ([]   (const int v) { return v * 2; }, 42, 21);
  test_proclaim_return_type<int&>  ([vp] () -> int& { return *vp; }, v);

  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    test_proclaim_return_type<int>(d_callable{}, 42);
  ),(
    test_proclaim_return_type<int>(h_callable{}, 42);
  ))


  // execution space annotations on lambda require --extended-lambda flag with nvrtc
#if !defined(TEST_COMPILER_NVRTC)
  NV_IF_TARGET(NV_IS_DEVICE, (
    test_proclaim_return_type<double>([]   __device__ { return 42.0; }, 42.0);
    test_proclaim_return_type<int>   ([]   __device__ (const int v) { return v * 2; }, 42, 21);
    test_proclaim_return_type<int&>  ([vp] __device__ () -> int& { return *vp; }, v);

    test_proclaim_return_type<double>([] __host__ __device__ { return 42.0; }, 42.0);
    test_proclaim_return_type<int>   ([] __host__ __device__ (const int v) { return v * 2; }, 42, 21);
    test_proclaim_return_type<int&>  ([vp] __host__ __device__ () -> int& { return *vp; }, v);
  ))

  // Ensure that we can always declare functions even on host
  auto f = cuda::proclaim_return_type<bool>([] __device__() { return false; });
  auto g = cuda::proclaim_return_type<bool>([f] __device__() { return f(); });

  unused(f);
  unused(g);
#endif // !TEST_COMPILER_NVRTC


  return 0;
}
