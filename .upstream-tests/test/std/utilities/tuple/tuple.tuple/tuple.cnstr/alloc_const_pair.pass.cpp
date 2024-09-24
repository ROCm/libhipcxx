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

// template <class Alloc, class U1, class U2>
//   tuple(allocator_arg_t, const Alloc& a, const pair<U1, U2>&);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/utility>
#include <hip/std/cassert>

#include "test_macros.h"
#include "allocators.h"
#include "../alloc_first.h"
#include "../alloc_last.h"

int main(int, char**)
{
    {
        typedef hip::std::pair<long, int> T0;
        typedef hip::std::tuple<long long, double> T1;
        T0 t0(2, 3);
        T1 t1(hip::std::allocator_arg, A1<int>(5), t0);
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1) == 3);
    }
    {
        typedef hip::std::pair<int, int> T0;
        typedef hip::std::tuple<alloc_first, double> T1;
        T0 t0(2, 3);
        alloc_first::allocator_constructed() = false;
        T1 t1(hip::std::allocator_arg, A1<int>(5), t0);
        assert(alloc_first::allocator_constructed());
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1) == 3);
    }
    {
        typedef hip::std::pair<int, int> T0;
        typedef hip::std::tuple<alloc_first, alloc_last> T1;
        T0 t0(2, 3);
        alloc_first::allocator_constructed() = false;
        alloc_last::allocator_constructed() = false;
        T1 t1(hip::std::allocator_arg, A1<int>(5), t0);
        assert(alloc_first::allocator_constructed());
        assert(alloc_last::allocator_constructed());
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1) == 3);
    }

  return 0;
}
