//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc, class... UTypes>
//   tuple(allocator_arg_t, const Alloc& a, tuple<UTypes...>&&);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"
#include "allocators.h"
#include "../alloc_first.h"
#include "../alloc_last.h"

struct B
{
    int id_;

    __host__ __device__ explicit B(int i) : id_(i) {}

    __host__ __device__ virtual ~B() {}
};

struct D
    : B
{
    __host__ __device__ explicit D(int i) : B(i) {}
};

struct Explicit {
  int value;
  __host__ __device__ explicit Explicit(int x) : value(x) {}
};

struct Implicit {
  int value;
  __host__ __device__ Implicit(int x) : value(x) {}
};

int main(int, char**)
{
    {
        typedef hip::std::tuple<int> T0;
        typedef hip::std::tuple<alloc_first> T1;
        T0 t0(2);
        alloc_first::allocator_constructed() = false;
        T1 t1(hip::std::allocator_arg, A1<int>(5), hip::std::move(t0));
        assert(alloc_first::allocator_constructed());
        assert(hip::std::get<0>(t1) == 2);
    }
    // hip::std::unique_ptr not supported
    // hip::std::allocator not supported
    /*
    {
        typedef hip::std::tuple<hip::std::unique_ptr<D>> T0;
        typedef hip::std::tuple<hip::std::unique_ptr<B>> T1;
        T0 t0(hip::std::unique_ptr<D>(new D(3)));
        T1 t1(hip::std::allocator_arg, A1<int>(5), hip::std::move(t0));
        assert(hip::std::get<0>(t1)->id_ == 3);
    }
    {
        typedef hip::std::tuple<int, hip::std::unique_ptr<D>> T0;
        typedef hip::std::tuple<alloc_first, hip::std::unique_ptr<B>> T1;
        T0 t0(2, hip::std::unique_ptr<D>(new D(3)));
        alloc_first::allocator_constructed() = false;
        T1 t1(hip::std::allocator_arg, A1<int>(5), hip::std::move(t0));
        assert(alloc_first::allocator_constructed());
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1)->id_ == 3);
    }
    {
        typedef hip::std::tuple<int, int, hip::std::unique_ptr<D>> T0;
        typedef hip::std::tuple<alloc_last, alloc_first, hip::std::unique_ptr<B>> T1;
        T0 t0(1, 2, hip::std::unique_ptr<D>(new D(3)));
        alloc_first::allocator_constructed() = false;
        alloc_last::allocator_constructed() = false;
        T1 t1(hip::std::allocator_arg, A1<int>(5), hip::std::move(t0));
        assert(alloc_first::allocator_constructed());
        assert(alloc_last::allocator_constructed());
        assert(hip::std::get<0>(t1) == 1);
        assert(hip::std::get<1>(t1) == 2);
        assert(hip::std::get<2>(t1)->id_ == 3);
    }
    {
        hip::std::tuple<int> t1(42);
        hip::std::tuple<Explicit> t2{hip::std::allocator_arg, hip::std::allocator<void>{}, hip::std::move(t1)};
        assert(hip::std::get<0>(t2).value == 42);
    }
    {
        hip::std::tuple<int> t1(42);
        hip::std::tuple<Implicit> t2 = {hip::std::allocator_arg, hip::std::allocator<void>{}, hip::std::move(t1)};
        assert(hip::std::get<0>(t2).value == 42);
    }
    */

  return 0;
}
