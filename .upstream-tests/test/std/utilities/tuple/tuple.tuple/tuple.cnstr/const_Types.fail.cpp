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

// explicit tuple(const T&...);

// UNSUPPORTED: c++98, c++03 
// UNSUPPORTED: nvrtc

#include <hip/std/tuple>
#include <hip/std/cassert>

struct ExplicitCopy {
  __host__ __device__ ExplicitCopy(int) {}
  __host__ __device__ explicit ExplicitCopy(ExplicitCopy const&) {}
};

__host__ __device__ std::tuple<ExplicitCopy> const_explicit_copy() {
    const ExplicitCopy e(42);
    return {e};
    // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}


__host__ __device__ std::tuple<ExplicitCopy> non_const_explicit_copy() {
    ExplicitCopy e(42);
    return {e};
    // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

__host__ __device__ std::tuple<ExplicitCopy> const_explicit_copy_no_brace() {
    const ExplicitCopy e(42);
    return e;
    // expected-error@-1 {{no viable conversion}}
}

int main(int, char**)
{

  return 0;
}
