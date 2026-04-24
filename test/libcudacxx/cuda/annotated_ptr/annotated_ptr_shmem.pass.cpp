//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <<<<<<< OLD CODE from 7011c55229 (9e9eeeb439) - COMMENTED OUT
// // <<<<<<< OLD CODE from 2ceb15d672 (5214850b75) - COMMENTED OUT
// // // MIT License
// // //
// // // Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// // //
// // // Permission is hereby granted, free of charge, to any person obtaining a copy
// // // of this software and associated documentation files (the "Software"), to deal
// // // in the Software without restriction, including without limitation the rights
// // // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// // // copies of the Software, and to permit persons to whom the Software is
// // // furnished to do so, subject to the following conditions:
// // //
// // // The above copyright notice and this permission notice shall be included in all
// // // copies or substantial portions of the Software.
// // //
// // // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// // // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// // // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// // // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// // // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// // // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// // // SOFTWARE.
// //
// // // UNSUPPORTED: nvrtc, hiprtc
// //
// // =======
// =======
// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

// >>>>>>> END NEW CODE (9e9eeeb439)
#include "test_macros.h"
// >>>>>>> END NEW CODE (5214850b75)
#include "utils.h"

template <typename T, typename U>
TEST_DEVICE_FUNC __noinline__ void shared_mem_test_dev()
{
  T* smem  = shared_alloc<T, 128>();
  smem[10] = 42;

  cuda::annotated_ptr<U, cuda::access_property::shared> p{smem + 10};

  assert(*p == 42);
}

TEST_DEVICE_FUNC __noinline__ void test_all()
{
  shared_mem_test_dev<int, int>();
  shared_mem_test_dev<int, const int>();
  shared_mem_test_dev<int, volatile int>();
  shared_mem_test_dev<int, const volatile int>();
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_IS_DEVICE, (test_all();))
  return 0;
}
