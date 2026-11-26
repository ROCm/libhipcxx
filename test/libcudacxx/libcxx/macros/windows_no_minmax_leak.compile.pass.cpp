//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Including libhipcxx public headers must not leak the min/max macros that <windows.h> defines
// (unless the user did not request NOMINMAX and included windows.h themselves). Otherwise unrelated
// user code such as std::min(...) or numeric_limits<T>::max() fails to compile.

// The test harness passes -DNOMINMAX on Windows. Undefine it so that the libhipcxx headers have to
// protect themselves.
#undef NOMINMAX

// <cuda/std/atomic> reaches <cuda/std/__thread/threading_support.h> (via __atomic/wait/polling.h), which
// includes <windows.h> on Windows. <cuda/std/latch> and <cuda/std/semaphore> are not supported on AMD
// hardware (they #error), so they cannot be used here.
#include <cuda/std/atomic>

#if defined(min)
#  error "libhipcxx public headers must not leak the min macro from <windows.h>"
#endif
#if defined(max)
#  error "libhipcxx public headers must not leak the max macro from <windows.h>"
#endif
// The includer's NOMINMAX state must be restored (it was not defined before the includes).
#if defined(NOMINMAX)
#  error "libhipcxx public headers must not leave NOMINMAX defined"
#endif

#include <algorithm>
#include <limits>

int main(int, char**)
{
  // These would fail to parse if min/max were function-like macros.
  const int smaller = std::min(1, 2);
  const int larger  = std::max(1, 2);
  static_assert(std::numeric_limits<int>::max() > 0, "");
  return smaller < larger ? 0 : 1;
}
