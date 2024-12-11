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

// time_point

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator==(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator!=(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// time_points with different clocks should not compare

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC
// UNSUPPORTED: nvrtc

#include <hip/std/chrono>

#include "../../clock.h"

int main(int, char**)
{
    typedef hip::std::chrono::system_clock Clock1;
    typedef Clock                     Clock2;
    typedef hip::std::chrono::milliseconds Duration1;
    typedef hip::std::chrono::microseconds Duration2;
    typedef hip::std::chrono::time_point<Clock1, Duration1> T1;
    typedef hip::std::chrono::time_point<Clock2, Duration2> T2;

    T1 t1(Duration1(3));
    T2 t2(Duration2(3000));
    t1 == t2;

  return 0;
}
