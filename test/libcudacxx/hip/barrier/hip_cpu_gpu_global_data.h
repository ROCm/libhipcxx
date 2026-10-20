//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Common managed-memory layout for CPU/GPU barrier tests.
//
// The including file must declare `barrier_t` before this header:
//   using barrier_t = hip::barrier<hip::thread_scope_system>;          // no completion
//   using barrier_t = hip::barrier<hip::thread_scope_system, void(*)()>; // with completion
//
// Tests needing a different data layout should define their own GlobalData
// instead of including this header. barrier_t::barrier must remain the first member.

#ifndef LIBHIPCXX_HIP_CPU_GPU_GLOBAL_DATA
#define LIBHIPCXX_HIP_CPU_GPU_GLOBAL_DATA

namespace hip_test {

struct GlobalData
{
  barrier_t barrier;
  int       x;
};

} // namespace hip_test

#endif // LIBHIPCXX_HIP_CPU_GPU_GLOBAL_DATA
