// -*- C++ -*-
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
// -*- C++ -*-

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: nvrtc

// Some early versions (cl.exe 14.16 / VC141) do not identify correct constructors
// UNSUPPORTED: msvc

// <cuda/std/tuple>

// template <class TupleLike> tuple(TupleLike&&); // libc++ extension

// See llvm.org/PR31384
#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
__device__ int count = 0;
#else
int count = 0;
#endif

struct Explicit {
  Explicit() = default;
  __host__ __device__ explicit Explicit(int) {}
};

struct Implicit {
  Implicit() = default;
  __host__ __device__ Implicit(int) {}
};

template<class T>
struct Derived : hip::std::tuple<T> {
  using hip::std::tuple<T>::tuple;
  template<class U>
  __host__ __device__ operator hip::std::tuple<U>() && { ++count; return {}; }
};


template<class T>
struct ExplicitDerived : hip::std::tuple<T> {
  using hip::std::tuple<T>::tuple;
  template<class U>
  __host__ __device__ explicit operator hip::std::tuple<U>() && { ++count; return {}; }
};

int main(int, char**) {
  {
    hip::std::tuple<Explicit> foo = Derived<int>{42}; ((void)foo);
    assert(count == 1);
    hip::std::tuple<Explicit> bar(Derived<int>{42}); ((void)bar);
    assert(count == 2);
  }
  count = 0;
  {
    hip::std::tuple<Implicit> foo = Derived<int>{42}; ((void)foo);
    assert(count == 1);
    hip::std::tuple<Implicit> bar(Derived<int>{42}); ((void)bar);
    assert(count == 2);
  }
  count = 0;
  {
    static_assert(!hip::std::is_convertible<
        ExplicitDerived<int>, hip::std::tuple<Explicit>>::value, "");
    hip::std::tuple<Explicit> bar(ExplicitDerived<int>{42}); ((void)bar);
    assert(count == 1);
  }
  count = 0;
  {
    // FIXME: Libc++ incorrectly rejects this code.
#ifndef _LIBCUDACXX_VERSION
    hip::std::tuple<Implicit> foo = ExplicitDerived<int>{42}; ((void)foo);
    static_assert(hip::std::is_convertible<
        ExplicitDerived<int>, hip::std::tuple<Implicit>>::value,
        "correct STLs accept this");
#else
    static_assert(!hip::std::is_convertible<
        ExplicitDerived<int>, hip::std::tuple<Implicit>>::value,
        "libc++ incorrectly rejects this");
#endif
    assert(count == 0);
    hip::std::tuple<Implicit> bar(ExplicitDerived<int>{42}); ((void)bar);
    assert(count == 1);
  }
  count = 0;


  return 0;
}
