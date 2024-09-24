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
//   tuple(allocator_arg_t, const Alloc& a, UTypes&&...);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "allocators.h"
#include "../alloc_first.h"
#include "../alloc_last.h"

template <class T = void>
struct DefaultCtorBlowsUp {
  __host__ __device__ constexpr DefaultCtorBlowsUp() {
      static_assert(!hip::std::is_same<T, T>::value, "Default Ctor instantiated");
  }

  __host__ __device__ explicit constexpr DefaultCtorBlowsUp(int x) : value(x) {}

  int value;
};


struct DerivedFromAllocArgT : hip::std::allocator_arg_t {};


// Make sure the _Up... constructor SFINAEs out when the number of initializers
// is less that the number of elements in the tuple. Previously libc++ would
// offer these constructers as an extension but they broke conforming code.
__host__ __device__ void test_uses_allocator_sfinae_evaluation()
{
    using BadDefault = DefaultCtorBlowsUp<>;
    {
        typedef hip::std::tuple<MoveOnly, MoveOnly, BadDefault> Tuple;

        static_assert(!hip::std::is_constructible<
            Tuple,
            hip::std::allocator_arg_t, A1<int>, MoveOnly
        >::value, "");

        static_assert(hip::std::is_constructible<
            Tuple,
            hip::std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly, BadDefault
        >::value, "");
    }
    {
        typedef hip::std::tuple<MoveOnly, MoveOnly, BadDefault, BadDefault> Tuple;

        static_assert(!hip::std::is_constructible<
            Tuple,
            hip::std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly
        >::value, "");

        static_assert(hip::std::is_constructible<
            Tuple,
            hip::std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly, BadDefault, BadDefault
        >::value, "");
    }
}

struct Explicit {
  int value;
  __host__ __device__ explicit Explicit(int x) : value(x) {}
};

int main(int, char**)
{
    // hip::std::allocator not supported
    /*
    {
        hip::std::tuple<Explicit> t{hip::std::allocator_arg, hip::std::allocator<void>{}, 42};
        assert(hip::std::get<0>(t).value == 42);
    }
    */
    {
        hip::std::tuple<MoveOnly> t(hip::std::allocator_arg, A1<int>(), MoveOnly(0));
        assert(hip::std::get<0>(t) == 0);
    }
    {
        using T = DefaultCtorBlowsUp<>;
        hip::std::tuple<T> t(hip::std::allocator_arg, A1<int>(), T(42));
        assert(hip::std::get<0>(t).value == 42);
    }
    {
        hip::std::tuple<MoveOnly, MoveOnly> t(hip::std::allocator_arg, A1<int>(),
                                         MoveOnly(0), MoveOnly(1));
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == 1);
    }
    {
        using T = DefaultCtorBlowsUp<>;
        hip::std::tuple<T, T> t(hip::std::allocator_arg, A1<int>(), T(42), T(43));
        assert(hip::std::get<0>(t).value == 42);
        assert(hip::std::get<1>(t).value == 43);
    }
    {
        hip::std::tuple<MoveOnly, MoveOnly, MoveOnly> t(hip::std::allocator_arg, A1<int>(),
                                                   MoveOnly(0),
                                                   1, 2);
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == 1);
        assert(hip::std::get<2>(t) == 2);
    }
    {
        using T = DefaultCtorBlowsUp<>;
        hip::std::tuple<T, T, T> t(hip::std::allocator_arg, A1<int>(), T(1), T(2), T(3));
        assert(hip::std::get<0>(t).value == 1);
        assert(hip::std::get<1>(t).value == 2);
        assert(hip::std::get<2>(t).value == 3);
    }
    {
        alloc_first::allocator_constructed() = false;
        alloc_last::allocator_constructed() = false;
        hip::std::tuple<int, alloc_first, alloc_last> t(hip::std::allocator_arg,
                                                   A1<int>(5), 1, 2, 3);
        assert(hip::std::get<0>(t) == 1);
        assert(alloc_first::allocator_constructed());
        assert(hip::std::get<1>(t) == alloc_first(2));
        assert(alloc_last::allocator_constructed());
        assert(hip::std::get<2>(t) == alloc_last(3));
    }
    {
        // Check that uses-allocator construction is still selected when
        // given a tag type that derives from allocator_arg_t.
        DerivedFromAllocArgT tag;
        alloc_first::allocator_constructed() = false;
        alloc_last::allocator_constructed() = false;
        hip::std::tuple<int, alloc_first, alloc_last> t(tag,
                                                   A1<int>(5), 1, 2, 3);
        assert(hip::std::get<0>(t) == 1);
        assert(alloc_first::allocator_constructed());
        assert(hip::std::get<1>(t) == alloc_first(2));
        assert(alloc_last::allocator_constructed());
        assert(hip::std::get<2>(t) == alloc_last(3));
    }
    // Stress test the SFINAE on the uses-allocator constructors and
    // ensure that the "reduced-arity-initialization" extension is not offered
    // for these constructors.
    test_uses_allocator_sfinae_evaluation();

  return 0;
}
