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

// T *data();

#include <hip/std/array>
#include <hip/std/cassert>
#include <hip/std/cstddef>       // for hip::std::max_align_t

#include "test_macros.h"

// hip::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

struct NoDefault {
  __host__ __device__ NoDefault(int) {}
};


int main(int, char**)
{
    {
        typedef double T;
        typedef hip::std::array<T, 3> C;
        C c = {1, 2, 3.5};
        T* p = c.data();
        assert(p[0] == 1);
        assert(p[1] == 2);
        assert(p[2] == 3.5);
    }
    {
        typedef double T;
        typedef hip::std::array<T, 0> C;
        C c = {};
        T* p = c.data();
        LIBCPP_ASSERT(p != nullptr);
    }
    {
      typedef double T;
      typedef hip::std::array<const T, 0> C;
      C c = {{}};
      const T* p = c.data();
      static_assert((hip::std::is_same<decltype(c.data()), const T*>::value), "");
      LIBCPP_ASSERT(p != nullptr);
    }
  {
      typedef hip::std::max_align_t T;
      typedef hip::std::array<T, 0> C;
      const C c = {};
      const T* p = c.data();
      LIBCPP_ASSERT(p != nullptr);
      hip::std::uintptr_t pint = reinterpret_cast<hip::std::uintptr_t>(p);
      assert(pint % TEST_ALIGNOF(hip::std::max_align_t) == 0);
    }
    {
      typedef NoDefault T;
      typedef hip::std::array<T, 0> C;
      C c = {};
      T* p = c.data();
      LIBCPP_ASSERT(p != nullptr);
    }

  return 0;
}
