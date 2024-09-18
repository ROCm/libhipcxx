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

// template <class Alloc>
//   tuple(allocator_arg_t, const Alloc& a, const Types&...);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"
#include "allocators.h"
#include "../alloc_first.h"
#include "../alloc_last.h"

struct ImplicitCopy {
  __host__ __device__ explicit ImplicitCopy(int) {}
  __host__ __device__ ImplicitCopy(ImplicitCopy const&) {}
};

// hip::std::allocator not supported
/*
// Test that tuple(hip::std::allocator_arg, Alloc, Types const&...) allows implicit
// copy conversions in return value expressions.
hip::std::tuple<ImplicitCopy> testImplicitCopy1() {
    ImplicitCopy i(42);
    return {hip::std::allocator_arg, hip::std::allocator<void>{}, i};
}

hip::std::tuple<ImplicitCopy> testImplicitCopy2() {
    const ImplicitCopy i(42);
    return {hip::std::allocator_arg, hip::std::allocator<void>{}, i};
}
*/

int main(int, char**)
{
    // Static initialization not supported on GPUs
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed() = false;
    // hip::std::allocator not supported
    /*
    {
        // check that the literal '0' can implicitly initialize a stored pointer.
        hip::std::tuple<int*> t = {hip::std::allocator_arg, hip::std::allocator<void>{}, 0};
    }
    */
    {
        hip::std::tuple<int> t(hip::std::allocator_arg, A1<int>(), 3);
        assert(hip::std::get<0>(t) == 3);
    }
    {
        assert(!alloc_first::allocator_constructed());
        hip::std::tuple<alloc_first> t(hip::std::allocator_arg, A1<int>(5), alloc_first(3));
        assert(alloc_first::allocator_constructed());
        assert(hip::std::get<0>(t) == alloc_first(3));
    }
    {
        assert(!alloc_last::allocator_constructed());
        hip::std::tuple<alloc_last> t(hip::std::allocator_arg, A1<int>(5), alloc_last(3));
        assert(alloc_last::allocator_constructed());
        assert(hip::std::get<0>(t) == alloc_last(3));
    }
    {
        alloc_first::allocator_constructed() = false;
        hip::std::tuple<int, alloc_first> t(hip::std::allocator_arg, A1<int>(5),
                                       10, alloc_first(15));
        assert(hip::std::get<0>(t) == 10);
        assert(alloc_first::allocator_constructed());
        assert(hip::std::get<1>(t) == alloc_first(15));
    }
    {
        alloc_first::allocator_constructed() = false;
        alloc_last::allocator_constructed() = false;
        hip::std::tuple<int, alloc_first, alloc_last> t(hip::std::allocator_arg,
                                                   A1<int>(5), 1, alloc_first(2),
                                                   alloc_last(3));
        assert(hip::std::get<0>(t) == 1);
        assert(alloc_first::allocator_constructed());
        assert(hip::std::get<1>(t) == alloc_first(2));
        assert(alloc_last::allocator_constructed());
        assert(hip::std::get<2>(t) == alloc_last(3));
    }
    {
        alloc_first::allocator_constructed() = false;
        alloc_last::allocator_constructed() = false;
        hip::std::tuple<int, alloc_first, alloc_last> t(hip::std::allocator_arg,
                                                   A2<int>(5), 1, alloc_first(2),
                                                   alloc_last(3));
        assert(hip::std::get<0>(t) == 1);
        assert(!alloc_first::allocator_constructed());
        assert(hip::std::get<1>(t) == alloc_first(2));
        assert(!alloc_last::allocator_constructed());
        assert(hip::std::get<2>(t) == alloc_last(3));
    }

  return 0;
}
