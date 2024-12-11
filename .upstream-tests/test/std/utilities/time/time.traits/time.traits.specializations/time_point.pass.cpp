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

// template <class Clock, class Duration1, class Duration2>
// struct common_type<chrono::time_point<Clock, Duration1>, chrono::time_point<Clock, Duration2>>
// {
//     typedef chrono::time_point<Clock, typename common_type<Duration1, Duration2>::type> type;
// };

#include <hip/std/chrono>

template <class D1, class D2, class De>
__host__ __device__
void
test()
{
    typedef hip::std::chrono::system_clock C;
    typedef hip::std::chrono::time_point<C, D1> T1;
    typedef hip::std::chrono::time_point<C, D2> T2;
    typedef hip::std::chrono::time_point<C, De> Te;
    typedef typename hip::std::common_type<T1, T2>::type Tc;
    static_assert((hip::std::is_same<Tc, Te>::value), "");
}

int main(int, char**)
{
    test<hip::std::chrono::duration<int, hip::std::ratio<1, 100> >,
         hip::std::chrono::duration<long, hip::std::ratio<1, 1000> >,
         hip::std::chrono::duration<long, hip::std::ratio<1, 1000> > >();
    test<hip::std::chrono::duration<long, hip::std::ratio<1, 100> >,
         hip::std::chrono::duration<int, hip::std::ratio<1, 1000> >,
         hip::std::chrono::duration<long, hip::std::ratio<1, 1000> > >();
    test<hip::std::chrono::duration<char, hip::std::ratio<1, 30> >,
         hip::std::chrono::duration<short, hip::std::ratio<1, 1000> >,
         hip::std::chrono::duration<int, hip::std::ratio<1, 3000> > >();
    test<hip::std::chrono::duration<double, hip::std::ratio<21, 1> >,
         hip::std::chrono::duration<short, hip::std::ratio<15, 1> >,
         hip::std::chrono::duration<double, hip::std::ratio<3, 1> > >();

  return 0;
}
