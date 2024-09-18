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

// <cuda/std/array>

// template <class T, size_t N> void swap(array<T,N>& x, array<T,N>& y);

#include <hip/std/array>
#include <hip/std/cassert>

#include "test_macros.h"
// hip::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

struct NonSwappable {
  __host__ __device__ NonSwappable() {}
private:
  __host__ __device__ NonSwappable(NonSwappable const&);
  __host__ __device__ NonSwappable& operator=(NonSwappable const&);
};

template <class Tp>
__host__ __device__ decltype(swap(hip::std::declval<Tp>(), hip::std::declval<Tp>()))
can_swap_imp(int);

template <class Tp>
__host__ __device__ hip::std::false_type can_swap_imp(...);

template <class Tp>
struct can_swap : hip::std::is_same<decltype(can_swap_imp<Tp>(0)), void> {};

int main(int, char**)
{
    {
        typedef double T;
        typedef hip::std::array<T, 3> C;
        C c1 = {1, 2, 3.5};
        C c2 = {4, 5, 6.5};
        swap(c1, c2);
        assert(c1.size() == 3);
        assert(c1[0] == 4);
        assert(c1[1] == 5);
        assert(c1[2] == 6.5);
        assert(c2.size() == 3);
        assert(c2[0] == 1);
        assert(c2[1] == 2);
        assert(c2[2] == 3.5);
    }
    {
        typedef double T;
        typedef hip::std::array<T, 0> C;
        C c1 = {};
        C c2 = {};
        swap(c1, c2);
        assert(c1.size() == 0);
        assert(c2.size() == 0);
    }
    {
        typedef NonSwappable T;
        typedef hip::std::array<T, 0> C0;
        static_assert(can_swap<C0&>::value, "");
        C0 l = {};
        C0 r = {};
        swap(l, r);
#if TEST_STD_VER >= 11
        static_assert(noexcept(swap(l, r)), "");
#endif
    }
#if TEST_STD_VER >= 11
    {
        // NonSwappable is still considered swappable in C++03 because there
        // is no access control SFINAE.
        typedef NonSwappable T;
        typedef hip::std::array<T, 42> C1;
        static_assert(!can_swap<C1&>::value, "");
    }
#endif

  return 0;
}
