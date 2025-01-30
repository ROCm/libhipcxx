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
// UNSUPPORTED: c++98, c++03, c++11, nvrtc
// UNSUPPORTED: apple-clang-9
// <chrono>

// template <class Duration> class hh_mm_ss;
//   If Duration is not an instance of duration, the program is ill-formed.

#include <cuda/std/chrono>
#include <string>
#include <cuda/std/cassert>
#include "test_macros.h"

struct A {};

int main(int, char**)
{
    cuda::std::chrono::hh_mm_ss<void> h0;              // expected-error-re@.*chrono:* {{{{(static_assert|static assertion)}} failed {{.*}} {{"?}}template parameter of hh_mm_ss must be a cuda::std::chrono::duration{{"?}}}}
    cuda::std::chrono::hh_mm_ss<int> h1;               // expected-error-re@.*chrono:* {{{{(static_assert|static assertion)}} failed {{.*}} {{"?}}template parameter of hh_mm_ss must be a cuda::std::chrono::duration{{"?}}}}
    cuda::std::chrono::hh_mm_ss<std::string> h2;       // expected-error-re@.*chrono:* {{{{(static_assert|static assertion)}} failed {{.*}} {{"?}}template parameter of hh_mm_ss must be a cuda::std::chrono::duration{{"?}}}}
    cuda::std::chrono::hh_mm_ss<A> h3;                 // expected-error-re@.*chrono:* {{{{(static_assert|static assertion)}} failed {{.*}} {{"?}}template parameter of hh_mm_ss must be a cuda::std::chrono::duration{{"?}}}}

    return 0;
}
