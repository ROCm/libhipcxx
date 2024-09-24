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

// UNSUPPORTED: c++98, c++03, c++11

// type_traits

// is_swappable

#include <hip/std/type_traits>
// NOTE: These headers are not currently supported by libcu++.
//#include <hip/std/utility>
//#include <hip/std/vector>
#include "test_macros.h"

namespace MyNS {

// Make the test types non-copyable so that generic hip::std::swap is not valid.
struct A {
  A(A const&) = delete;
  A& operator=(A const&) = delete;
};

struct B {
  B(B const&) = delete;
  B& operator=(B const&) = delete;
};

struct C {};
struct D {};

__host__ __device__
void swap(A&, A&) {}

__host__ __device__
void swap(A&, B&) {}
__host__ __device__
void swap(B&, A&) {}

__host__ __device__
void swap(A&, C&) {} // missing swap(C, A)
__host__ __device__
void swap(D&, C&) {}

struct M {
  M(M const&) = delete;
  M& operator=(M const&) = delete;
};

__host__ __device__
void swap(M&&, M&&) {}

struct DeletedSwap {
  __host__ __device__
  friend void swap(DeletedSwap&, DeletedSwap&) = delete;
};

} // namespace MyNS

namespace MyNS2 {

struct AmbiguousSwap {};

template <class T>
__host__ __device__
void swap(T&, T&) {}

} // end namespace MyNS2

int main(int, char**)
{
    using namespace MyNS;
    {
        // Test that is_swappable applies an lvalue reference to the type.
        static_assert(hip::std::is_swappable<A>::value, "");
        static_assert(hip::std::is_swappable<A&>::value, "");
        static_assert(!hip::std::is_swappable<M>::value, "");
        static_assert(!hip::std::is_swappable<M&&>::value, "");
    }
    static_assert(!hip::std::is_swappable<B>::value, "");
    static_assert(hip::std::is_swappable<C>::value, "");
    {
        // test non-referencable types
        static_assert(!hip::std::is_swappable<void>::value, "");
        static_assert(!hip::std::is_swappable<int() const>::value, "");
        static_assert(!hip::std::is_swappable<int() &>::value, "");
    }
    {
        // test that a deleted swap is correctly handled.
        static_assert(!hip::std::is_swappable<DeletedSwap>::value, "");
    }
    {
        // test that a swap with ambiguous overloads is handled correctly.
        static_assert(!hip::std::is_swappable<MyNS2::AmbiguousSwap>::value, "");
    }
    {
        // test for presence of is_swappable_v
        static_assert(hip::std::is_swappable_v<int>, "");
        static_assert(!hip::std::is_swappable_v<M>, "");
    }

  return 0;
}
