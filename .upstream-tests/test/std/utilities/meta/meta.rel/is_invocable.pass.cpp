//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

// XFAIL: hipcc
// UNSUPPORTED: c++98, c++03, c++11

// XFAIL: nvcc
// FIXME: Triage and fix this.

// type_traits

// is_invocable

// Most testing of is_invocable is done within the [meta.trans.other] result_of
// tests.

// Fn and all types in the template parameter pack ArgTypes shall be
//  complete types, cv void, or arrays of unknown bound.

#include <cuda/std/type_traits>
// NOTE: These headers are not currently supported by libcu++.
#include <cuda/std/functional>
#include <cuda/std/memory>
#include <cuda/std/vector>

#include "test_macros.h"

struct Tag {};
struct DerFromTag : Tag {};

struct Implicit {
  Implicit(int) {}
};

struct Explicit {
  explicit Explicit(int) {}
};

struct NotCallableWithInt {
  int operator()(int) = delete;
  int operator()(Tag) { return 42; }
};

struct Sink {
  template <class ...Args>
  void operator()(Args&&...) const {}
};

int main(int, char**) {
  using AbominableFunc = void(...) const;

  //  Non-callable things
  {
    static_assert(!cuda::std::is_invocable<void>::value, "");
    static_assert(!cuda::std::is_invocable<const void>::value, "");
    static_assert(!cuda::std::is_invocable<volatile void>::value, "");
    static_assert(!cuda::std::is_invocable<const volatile void>::value, "");
    static_assert(!cuda::std::is_invocable<cuda::std::nullptr_t>::value, "");
    static_assert(!cuda::std::is_invocable<int>::value, "");
    static_assert(!cuda::std::is_invocable<double>::value, "");

    static_assert(!cuda::std::is_invocable<int[]>::value, "");
    static_assert(!cuda::std::is_invocable<int[3]>::value, "");

    static_assert(!cuda::std::is_invocable<int*>::value, "");
    static_assert(!cuda::std::is_invocable<const int*>::value, "");
    static_assert(!cuda::std::is_invocable<int const*>::value, "");

    static_assert(!cuda::std::is_invocable<int&>::value, "");
    static_assert(!cuda::std::is_invocable<const int&>::value, "");
    static_assert(!cuda::std::is_invocable<int&&>::value, "");

    static_assert(!cuda::std::is_invocable<cuda::std::vector<int> >::value, "");
    static_assert(!cuda::std::is_invocable<cuda::std::vector<int*> >::value, "");
    static_assert(!cuda::std::is_invocable<cuda::std::vector<int**> >::value, "");

    static_assert(!cuda::std::is_invocable<AbominableFunc>::value, "");

    //  with parameters
    static_assert(!cuda::std::is_invocable<int, int>::value, "");
    static_assert(!cuda::std::is_invocable<int, double, float>::value, "");
    static_assert(!cuda::std::is_invocable<int, char, float, double>::value, "");
    static_assert(!cuda::std::is_invocable<Sink, AbominableFunc>::value, "");
    static_assert(!cuda::std::is_invocable<Sink, void>::value, "");
    static_assert(!cuda::std::is_invocable<Sink, const volatile void>::value,
                  "");


    static_assert(!cuda::std::is_invocable_r<int, void>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, const void>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, volatile void>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, const volatile void>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, cuda::std::nullptr_t>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, int>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, double>::value, "");

    static_assert(!cuda::std::is_invocable_r<int, int[]>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, int[3]>::value, "");

    static_assert(!cuda::std::is_invocable_r<int, int*>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, const int*>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, int const*>::value, "");

    static_assert(!cuda::std::is_invocable_r<int, int&>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, const int&>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, int&&>::value, "");

    static_assert(!cuda::std::is_invocable_r<int, cuda::std::vector<int> >::value, "");
    static_assert(!cuda::std::is_invocable_r<int, cuda::std::vector<int*> >::value, "");
    static_assert(!cuda::std::is_invocable_r<int, cuda::std::vector<int**> >::value, "");
    static_assert(!cuda::std::is_invocable_r<void, AbominableFunc>::value, "");

    //  with parameters
    static_assert(!cuda::std::is_invocable_r<int, int, int>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, int, double, float>::value, "");
    static_assert(!cuda::std::is_invocable_r<int, int, char, float, double>::value,
                  "");
    static_assert(!cuda::std::is_invocable_r<void, Sink, AbominableFunc>::value, "");
    static_assert(!cuda::std::is_invocable_r<void, Sink, void>::value, "");
    static_assert(!cuda::std::is_invocable_r<void, Sink, const volatile void>::value,
                  "");
  }
  {
    using Fn = int (Tag::*)(int);
    using RFn = int (Tag::*)(int)&&;
    // INVOKE bullet 1, 2 and 3
    {
      // Bullet 1
      static_assert(cuda::std::is_invocable<Fn, Tag&, int>::value, "");
      static_assert(cuda::std::is_invocable<Fn, DerFromTag&, int>::value, "");
      static_assert(cuda::std::is_invocable<RFn, Tag&&, int>::value, "");
      static_assert(!cuda::std::is_invocable<RFn, Tag&, int>::value, "");
      static_assert(!cuda::std::is_invocable<Fn, Tag&>::value, "");
      static_assert(!cuda::std::is_invocable<Fn, Tag const&, int>::value, "");
    }
    {
      // Bullet 2
      using T = cuda::std::reference_wrapper<Tag>;
      using DT = cuda::std::reference_wrapper<DerFromTag>;
      using CT = cuda::std::reference_wrapper<const Tag>;
      static_assert(cuda::std::is_invocable<Fn, T&, int>::value, "");
      static_assert(cuda::std::is_invocable<Fn, DT&, int>::value, "");
      static_assert(cuda::std::is_invocable<Fn, const T&, int>::value, "");
      static_assert(cuda::std::is_invocable<Fn, T&&, int>::value, "");
      static_assert(!cuda::std::is_invocable<Fn, CT&, int>::value, "");
      static_assert(!cuda::std::is_invocable<RFn, T, int>::value, "");
    }
    {
      // Bullet 3
      using T = Tag*;
      using DT = DerFromTag*;
      using CT = const Tag*;
      using ST = cuda::std::unique_ptr<Tag>;
      static_assert(cuda::std::is_invocable<Fn, T&, int>::value, "");
      static_assert(cuda::std::is_invocable<Fn, DT&, int>::value, "");
      static_assert(cuda::std::is_invocable<Fn, const T&, int>::value, "");
      static_assert(cuda::std::is_invocable<Fn, T&&, int>::value, "");
      static_assert(cuda::std::is_invocable<Fn, ST, int>::value, "");
      static_assert(!cuda::std::is_invocable<Fn, CT&, int>::value, "");
      static_assert(!cuda::std::is_invocable<RFn, T, int>::value, "");
    }
  }
  {
    // Bullets 4, 5 and 6
    using Fn = int(Tag::*);
    static_assert(!cuda::std::is_invocable<Fn>::value, "");
    {
      // Bullet 4
      static_assert(cuda::std::is_invocable<Fn, Tag&>::value, "");
      static_assert(cuda::std::is_invocable<Fn, DerFromTag&>::value, "");
      static_assert(cuda::std::is_invocable<Fn, Tag&&>::value, "");
      static_assert(cuda::std::is_invocable<Fn, Tag const&>::value, "");
    }
    {
      // Bullet 5
      using T = cuda::std::reference_wrapper<Tag>;
      using DT = cuda::std::reference_wrapper<DerFromTag>;
      using CT = cuda::std::reference_wrapper<const Tag>;
      static_assert(cuda::std::is_invocable<Fn, T&>::value, "");
      static_assert(cuda::std::is_invocable<Fn, DT&>::value, "");
      static_assert(cuda::std::is_invocable<Fn, const T&>::value, "");
      static_assert(cuda::std::is_invocable<Fn, T&&>::value, "");
      static_assert(cuda::std::is_invocable<Fn, CT&>::value, "");
    }
    {
      // Bullet 6
      using T = Tag*;
      using DT = DerFromTag*;
      using CT = const Tag*;
      using ST = cuda::std::unique_ptr<Tag>;
      static_assert(cuda::std::is_invocable<Fn, T&>::value, "");
      static_assert(cuda::std::is_invocable<Fn, DT&>::value, "");
      static_assert(cuda::std::is_invocable<Fn, const T&>::value, "");
      static_assert(cuda::std::is_invocable<Fn, T&&>::value, "");
      static_assert(cuda::std::is_invocable<Fn, ST>::value, "");
      static_assert(cuda::std::is_invocable<Fn, CT&>::value, "");
    }
  }
  { // INVOKE bullet 7
   {// Function pointer
    using Fp = void(*)(Tag&, int);
  static_assert(cuda::std::is_invocable<Fp, Tag&, int>::value, "");
  static_assert(cuda::std::is_invocable<Fp, DerFromTag&, int>::value, "");
  static_assert(!cuda::std::is_invocable<Fp, const Tag&, int>::value, "");
  static_assert(!cuda::std::is_invocable<Fp>::value, "");
  static_assert(!cuda::std::is_invocable<Fp, Tag&>::value, "");
}
{
  // Function reference
  using Fp = void (&)(Tag&, int);
  static_assert(cuda::std::is_invocable<Fp, Tag&, int>::value, "");
  static_assert(cuda::std::is_invocable<Fp, DerFromTag&, int>::value, "");
  static_assert(!cuda::std::is_invocable<Fp, const Tag&, int>::value, "");
  static_assert(!cuda::std::is_invocable<Fp>::value, "");
  static_assert(!cuda::std::is_invocable<Fp, Tag&>::value, "");
}
{
  // Function object
  using Fn = NotCallableWithInt;
  static_assert(cuda::std::is_invocable<Fn, Tag>::value, "");
  static_assert(!cuda::std::is_invocable<Fn, int>::value, "");
}
}
{
  // Check that the conversion to the return type is properly checked
  using Fn = int (*)();
  static_assert(cuda::std::is_invocable_r<Implicit, Fn>::value, "");
  static_assert(cuda::std::is_invocable_r<double, Fn>::value, "");
  static_assert(cuda::std::is_invocable_r<const volatile void, Fn>::value, "");
  static_assert(!cuda::std::is_invocable_r<Explicit, Fn>::value, "");
}
{
  // Check for is_invocable_v
  using Fn = void (*)();
  static_assert(cuda::std::is_invocable_v<Fn>, "");
  static_assert(!cuda::std::is_invocable_v<Fn, int>, "");
}
{
  // Check for is_invocable_r_v
  using Fn = void (*)();
  static_assert(cuda::std::is_invocable_r_v<void, Fn>, "");
  static_assert(!cuda::std::is_invocable_r_v<int, Fn>, "");
}
  return 0;
}
