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

// UNSUPPORTED: hipcc
// This test is including cuda/std/vector which does not exist.
// HIPCC gives the correct error: cuda/std/string file not found.
// However, other errors are expected, consider the expected-error. 
// We found _LIBCUDACXX_HAS_VECTOR guard in other tests that include the header, 
// indicating that the header is not implemented yet. 
// <cuda/std/array>

// UNSUPPORTED: nvrtc

// bool operator==(array<T, N> const&, array<T, N> const&);
// bool operator!=(array<T, N> const&, array<T, N> const&);
// bool operator<(array<T, N> const&, array<T, N> const&);
// bool operator<=(array<T, N> const&, array<T, N> const&);
// bool operator>(array<T, N> const&, array<T, N> const&);
// bool operator>=(array<T, N> const&, array<T, N> const&);


#include <hip/std/array>
#include <hip/std/vector>
#include <hip/std/cassert>

#include "test_macros.h"

// hip::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

template <class Array>
__host__ __device__
void test_compare(const Array& LHS, const Array& RHS) {
  typedef hip::std::vector<typename Array::value_type> Vector;
  const Vector LHSV(LHS.begin(), LHS.end());
  const Vector RHSV(RHS.begin(), RHS.end());
  assert((LHS == RHS) == (LHSV == RHSV));
  assert((LHS != RHS) == (LHSV != RHSV));
  assert((LHS < RHS) == (LHSV < RHSV));
  assert((LHS <= RHS) == (LHSV <= RHSV));
  assert((LHS > RHS) == (LHSV > RHSV));
  assert((LHS >= RHS) == (LHSV >= RHSV));
}

template <int Dummy> struct NoCompare {};

int main(int, char**)
{
  {
    typedef NoCompare<0> T;
    typedef hip::std::array<T, 3> C;
    C c1 = {{}};
    // expected-error@algorithm:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD (c1 == c1);
    TEST_IGNORE_NODISCARD (c1 < c1);
  }
  {
    typedef NoCompare<1> T;
    typedef hip::std::array<T, 3> C;
    C c1 = {{}};
    // expected-error@algorithm:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD (c1 != c1);
    TEST_IGNORE_NODISCARD (c1 > c1);
  }
  {
    typedef NoCompare<2> T;
    typedef hip::std::array<T, 0> C;
    C c1 = {{}};
    // expected-error@algorithm:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD (c1 == c1);
    TEST_IGNORE_NODISCARD (c1 < c1);
  }

  return 0;
}
