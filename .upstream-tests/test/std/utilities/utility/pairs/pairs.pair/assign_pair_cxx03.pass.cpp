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

// REQUIRES: c++98 || c++03

// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair const& p);

#include <hip/std/utility>
// cuda/std/memory not supported
// #include <hip/std/memory>
#include <hip/std/cassert>

#include "test_macros.h"

struct NonAssignable {
  __host__ __device__ NonAssignable() {}
private:
  __host__ __device__ NonAssignable& operator=(NonAssignable const&);
};

struct Incomplete;
extern Incomplete inc_obj;

int main(int, char**)
{
    {
    // Test that we don't constrain the assignment operator in C++03 mode.
    // Since we don't have access control SFINAE having pair evaluate SFINAE
    // may cause a hard error.
    typedef hip::std::pair<int, NonAssignable> P;
    static_assert(hip::std::is_copy_assignable<P>::value, "");
    }
    {
    typedef hip::std::pair<int, Incomplete&> P;
    static_assert(hip::std::is_copy_assignable<P>::value, "");
    P p(42, inc_obj);
    assert(&p.second == &inc_obj);
    }

  return 0;
}

struct Incomplete {};
Incomplete inc_obj;
