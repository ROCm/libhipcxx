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

// REQUIRES: c++11
// UNSUPPORTED: nvrtc

// <utility>

// Test that only the default constructor is constexpr in C++11

#include <hip/std/utility>
#include <hip/std/cassert>

struct ExplicitT {
  __host__ __device__ constexpr explicit ExplicitT(int x) : value(x) {}
  __host__ __device__ constexpr explicit ExplicitT(ExplicitT const& o) : value(o.value) {}
  int value;
};

struct ImplicitT {
  __host__ __device__ constexpr ImplicitT(int x) : value(x) {}
  __host__ __device__ constexpr ImplicitT(ImplicitT const& o) : value(o.value) {}
  int value;
};

int main(int, char**)
{
    {
        using P = hip::std::pair<int, int>;
        constexpr int x = 42;
        constexpr P default_p{};
        constexpr P copy_p(default_p);
        constexpr P const_U_V(x, x); // expected-error {{must be initialized by a constant expression}}
        constexpr P U_V(42, 101); // expected-error {{must be initialized by a constant expression}}
    }
    {
        using P = hip::std::pair<ExplicitT, ExplicitT>;
        constexpr hip::std::pair<int, int> other;
        constexpr ExplicitT e(99);
        constexpr P const_U_V(e, e); // expected-error {{must be initialized by a constant expression}}
        constexpr P U_V(42, 101); // expected-error {{must be initialized by a constant expression}}
        constexpr P pair_U_V(other); // expected-error {{must be initialized by a constant expression}}
    }
    {
        using P = hip::std::pair<ImplicitT, ImplicitT>;
        constexpr hip::std::pair<int, int> other;
        constexpr ImplicitT i = 99;
        constexpr P const_U_V = {i, i}; // expected-error {{must be initialized by a constant expression}}
        constexpr P U_V = {42, 101}; // expected-error {{must be initialized by a constant expression}}
        constexpr P pair_U_V = other; // expected-error {{must be initialized by a constant expression}}
    }

  return 0;
}
