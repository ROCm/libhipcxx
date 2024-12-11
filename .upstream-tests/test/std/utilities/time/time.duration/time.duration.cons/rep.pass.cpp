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

// <cuda/std/chrono>

// duration

// template <class Rep2>
//   explicit duration(const Rep2& r);

#include <hip/std/chrono>
#include <hip/std/cassert>

#include "test_macros.h"
#include "../../rep.h"

template <class D, class R>
__host__ __device__
void
test(R r)
{
    D d(r);
    assert(d.count() == r);
#if TEST_STD_VER >= 11
    constexpr D d2(R(2));
    static_assert(d2.count() == 2, "");
#endif
}

int main(int, char**)
{
    test<hip::std::chrono::duration<int> >(5);
    test<hip::std::chrono::duration<int, hip::std::ratio<3, 2> > >(5);
    test<hip::std::chrono::duration<Rep, hip::std::ratio<3, 2> > >(Rep(3));
    test<hip::std::chrono::duration<double, hip::std::ratio<2, 3> > >(5.5);

  return 0;
}
