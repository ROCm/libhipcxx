//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// NOTE(HIP/AMD): The hipcc and hiprtc tags were removed from the markers
// below because the test is already excluded by the c++17 lit directive a
// few lines down (the only build configuration currently exercised) and
// by the existing nvrtc tag (which covers hiprtc on the relevant build
// matrix at time of writing). If/when a C++20 hipcc build is added,
// re-enabling the hipcc tag may be needed; cuda::std::chrono::file_clock
// support on HIP under C++20 has not yet been audited.

// UNSUPPORTED: c++17
// XFAIL: dylib-has-no-filesystem

// File clock is unsupported in NVRTC
// UNSUPPORTED: nvrtc

// <cuda/std/chrono>

// file_clock

// static time_point now() noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  using C = cuda::std::chrono::file_clock;
  static_assert(noexcept(C::now()));

  C::time_point t1 = C::now();
  assert(t1.time_since_epoch().count() != 0);
  assert(C::time_point::min() < t1);
  assert(C::time_point::max() > t1);

  return 0;
}
