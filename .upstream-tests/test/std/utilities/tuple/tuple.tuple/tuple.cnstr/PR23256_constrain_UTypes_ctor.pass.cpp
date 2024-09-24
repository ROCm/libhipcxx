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
// XFAIL: msvc

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class ...UTypes>
//    EXPLICIT(...) tuple(UTypes&&...)

// Check that the UTypes... ctor is properly disabled before evaluating any
// SFINAE when the tuple-like copy/move ctor should *clearly* be selected
// instead. This happens 'sizeof...(UTypes) == 1' and the first element of
// 'UTypes...' is an instance of the tuple itself. See PR23256.

#include <hip/std/tuple>
#include <hip/std/type_traits>

#include "test_macros.h"


struct UnconstrainedCtor {
  int value_;

  __host__ __device__ UnconstrainedCtor() : value_(0) {}

  // Blows up when instantiated for any type other than int. Because the ctor
  // is constexpr it is instantiated by 'is_constructible' and 'is_convertible'
  // for Clang based compilers. GCC does not instantiate the ctor body
  // but it does instantiate the noexcept specifier and it will blow up there.
  template <typename T>
  __host__ __device__ constexpr UnconstrainedCtor(T value) noexcept(noexcept(value_ = value))
      : value_(static_cast<int>(value))
  {
      static_assert(hip::std::is_same<int, T>::value, "");
  }
};

struct ExplicitUnconstrainedCtor {
  int value_;

  __host__ __device__ ExplicitUnconstrainedCtor() : value_(0) {}

  template <typename T>
  __host__ __device__ constexpr explicit ExplicitUnconstrainedCtor(T value)
    noexcept(noexcept(value_ = value))
      : value_(static_cast<int>(value))
  {
      static_assert(hip::std::is_same<int, T>::value, "");
  }

};

int main(int, char**) {
    typedef UnconstrainedCtor A;
    typedef ExplicitUnconstrainedCtor ExplicitA;
    {
        static_assert(hip::std::is_copy_constructible<hip::std::tuple<A>>::value, "");
        static_assert(hip::std::is_move_constructible<hip::std::tuple<A>>::value, "");
        static_assert(hip::std::is_copy_constructible<hip::std::tuple<ExplicitA>>::value, "");
        static_assert(hip::std::is_move_constructible<hip::std::tuple<ExplicitA>>::value, "");
    }
    // hip::std::allocator not supported
    /*
    {
        static_assert(hip::std::is_constructible<
            hip::std::tuple<A>,
            hip::std::allocator_arg_t, hip::std::allocator<void>,
            hip::std::tuple<A> const&
        >::value, "");
        static_assert(hip::std::is_constructible<
            hip::std::tuple<A>,
            hip::std::allocator_arg_t, hip::std::allocator<void>,
            hip::std::tuple<A> &&
        >::value, "");
        static_assert(hip::std::is_constructible<
            hip::std::tuple<ExplicitA>,
            hip::std::allocator_arg_t, hip::std::allocator<void>,
            hip::std::tuple<ExplicitA> const&
        >::value, "");
        static_assert(hip::std::is_constructible<
            hip::std::tuple<ExplicitA>,
            hip::std::allocator_arg_t, hip::std::allocator<void>,
            hip::std::tuple<ExplicitA> &&
        >::value, "");
    }
    */
    {
        hip::std::tuple<A&&> t(hip::std::forward_as_tuple(A{}));
        ((void)t);
        hip::std::tuple<ExplicitA&&> t2(hip::std::forward_as_tuple(ExplicitA{}));
        ((void)t2);
    }

  return 0;
}
