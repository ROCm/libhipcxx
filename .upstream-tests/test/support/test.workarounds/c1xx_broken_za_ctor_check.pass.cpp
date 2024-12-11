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

// UNSUPPORTED: c++98, c++03

// Verify TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK.

#include <hip/std/type_traits>

#include "test_macros.h"
#include "test_workarounds.h"

struct X {
    __host__ __device__
    X(int) {}

    X(X&&) = default;
    X& operator=(X&&) = default;

private:
    X(const X&) = default;
    X& operator=(const X&) = default;
};

__host__ __device__
void PushFront(X&&) {}

template<class T = int>
__host__ __device__
auto test(int) -> decltype(PushFront(hip::std::declval<T>()), hip::std::true_type{});
__host__ __device__
auto test(long) -> hip::std::false_type;

int main(int, char**) {
#if defined(TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK)
    static_assert(!decltype(test(0))::value, "");
#else
    static_assert(decltype(test(0))::value, "");
#endif

  return 0;
}
