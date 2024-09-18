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

// template <size_t I, class T, size_t N> const T&& get(const array<T, N>&& a);

// UNSUPPORTED: c++98, c++03

#include <hip/std/array>
#if defined(_LIBCUDACXX_HAS_MEMORY)
#include <hip/std/memory>
#endif
#include <hip/std/type_traits>
#include <hip/std/utility>
#include <hip/std/cassert>

#include "test_macros.h"

// hip::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main(int, char**)
{

#if defined(_LIBCUDACXX_HAS_MEMORY)
    {
    typedef hip::std::unique_ptr<double> T;
    typedef hip::std::array<T, 1> C;
    const C c = {hip::std::unique_ptr<double>(new double(3.5))};
    static_assert(hip::std::is_same<const T&&, decltype(hip::std::get<0>(hip::std::move(c)))>::value, "");
    static_assert(noexcept(hip::std::get<0>(hip::std::move(c))), "");
    const T&& t = hip::std::get<0>(hip::std::move(c));
    assert(*t == 3.5);
    }
#endif

#if TEST_STD_VER > 11
    {
    typedef double T;
    typedef hip::std::array<T, 3> C;
    constexpr const C c = {1, 2, 3.5};
    static_assert(hip::std::get<0>(hip::std::move(c)) == 1, "");
    static_assert(hip::std::get<1>(hip::std::move(c)) == 2, "");
    static_assert(hip::std::get<2>(hip::std::move(c)) == 3.5, "");
    }
#endif

  return 0;
}
