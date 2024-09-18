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

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   explicit(see-below) tuple(allocator_arg_t, const Alloc& a);

// NOTE: this constructor does not currently support tags derived from
// allocator_arg_t because libc++ has to deduce the parameter as a template
// argument. See PR27684 (https://bugs.llvm.org/show_bug.cgi?id=27684)

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"
#include "DefaultOnly.h"
#include "allocators.h"
#include "../alloc_first.h"
#include "../alloc_last.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

template <class T = void>
struct NonDefaultConstructible {
  __host__ __device__ constexpr NonDefaultConstructible() {
      static_assert(!hip::std::is_same<T, T>::value, "Default Ctor instantiated");
  }

  __host__ __device__ explicit constexpr NonDefaultConstructible(int) {}
};


struct DerivedFromAllocArgT : hip::std::allocator_arg_t {};

int main(int, char**)
{
    DefaultOnly::count() = 0;
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed() = false;
    {
        hip::std::tuple<> t(hip::std::allocator_arg, A1<int>());
        unused(t);
    }
    {
        hip::std::tuple<int> t(hip::std::allocator_arg, A1<int>());
        assert(hip::std::get<0>(t) == 0);
    }
    {
        hip::std::tuple<DefaultOnly> t(hip::std::allocator_arg, A1<int>());
        assert(hip::std::get<0>(t) == DefaultOnly());
    }
    {
        assert(!alloc_first::allocator_constructed());
        hip::std::tuple<alloc_first> t(hip::std::allocator_arg, A1<int>(5));
        assert(alloc_first::allocator_constructed());
        assert(hip::std::get<0>(t) == alloc_first());
    }
    {
        assert(!alloc_last::allocator_constructed());
        hip::std::tuple<alloc_last> t(hip::std::allocator_arg, A1<int>(5));
        assert(alloc_last::allocator_constructed());
        assert(hip::std::get<0>(t) == alloc_last());
    }
    {
        alloc_first::allocator_constructed() = false;
        hip::std::tuple<DefaultOnly, alloc_first> t(hip::std::allocator_arg, A1<int>(5));
        assert(hip::std::get<0>(t) == DefaultOnly());
        assert(alloc_first::allocator_constructed());
        assert(hip::std::get<1>(t) == alloc_first());
    }
    {
        alloc_first::allocator_constructed() = false;
        alloc_last::allocator_constructed() = false;
        hip::std::tuple<DefaultOnly, alloc_first, alloc_last> t(hip::std::allocator_arg,
                                                           A1<int>(5));
        assert(hip::std::get<0>(t) == DefaultOnly());
        assert(alloc_first::allocator_constructed());
        assert(hip::std::get<1>(t) == alloc_first());
        assert(alloc_last::allocator_constructed());
        assert(hip::std::get<2>(t) == alloc_last());
    }
    {
        alloc_first::allocator_constructed() = false;
        alloc_last::allocator_constructed() = false;
        hip::std::tuple<DefaultOnly, alloc_first, alloc_last> t(hip::std::allocator_arg,
                                                           A2<int>(5));
        assert(hip::std::get<0>(t) == DefaultOnly());
        assert(!alloc_first::allocator_constructed());
        assert(hip::std::get<1>(t) == alloc_first());
        assert(!alloc_last::allocator_constructed());
        assert(hip::std::get<2>(t) == alloc_last());
    }
    /*
    {
        // Test that the uses-allocator default constructor does not evaluate
        // its SFINAE when it otherwise shouldn't be selected. Do this by
        // using 'NonDefaultConstructible' which will cause a compile error
        // if hip::std::is_default_constructible is evaluated on it.
        using T = NonDefaultConstructible<>;
        T v(42);
        hip::std::tuple<T, T> t(v, v);
        unused(t);
        hip::std::tuple<T, T> t2(42, 42);
        unused(t2);
    }
    */
  return 0;
}
