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
// // MIT License
// //
// // Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// //
// // Permission is hereby granted, free of charge, to any person obtaining a copy
// // of this software and associated documentation files (the "Software"), to deal
// // in the Software without restriction, including without limitation the rights
// // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// // copies of the Software, and to permit persons to whom the Software is
// // furnished to do so, subject to the following conditions:
// //
// // The above copyright notice and this permission notice shall be included in all
// // copies or substantial portions of the Software.
// //
// // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// // SOFTWARE.
//
// // UNSUPPORTED: hipcc, hiprtc
// =======
// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code
// error: accessing gridDim/blockDim/blockIdx/threadIdx/warpSize is unsupported in tile code
// >>>>>>> END NEW CODE (9e9eeeb439)

// UNSUPPORTED: pre-sm-70

// clang-cuda < 20 errors out with "fatal error: error in backend: Cannot cast between two non-generic address spaces"
// XFAIL: clang-14 && !nvcc
// XFAIL: clang-15 && !nvcc
// XFAIL: clang-16 && !nvcc
// XFAIL: clang-17 && !nvcc
// XFAIL: clang-18 && !nvcc
// XFAIL: clang-19 && !nvcc

#include "memcpy_async.h"

int main(int argc, char** argv)
{
  test_select_source<int32_t>();

  return 0;
}
