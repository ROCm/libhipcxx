//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <<<<<<< OLD CODE from 8feaa3a149 (c75f819baf) - COMMENTED OUT
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
// =======
// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS
// >>>>>>> END NEW CODE (c75f819baf)

#include <cuda/std/__cccl/extended_data_types.h>

#include "test_macros.h"

int main(int, char**)
{
// <<<<<<< OLD CODE from 8feaa3a149 (c75f819baf) - COMMENTED OUT
//   // Note(HIP/AMD): There is no official support for __float128 but it is already
//   // available for the compiler. So we have to use the else case.
// #if !_CCCL_HAS_FLOAT128() && !defined(__HIP_PLATFORM_AMD__)
//   __float128 x4 = __float128(3.14) + __float128(3.14);
//   unused(x4);
// =======
#if !_CCCL_HAS_FLOAT128()
  [[maybe_unused]] __float128 x4 = 3.14q + 3.14Q;
// >>>>>>> END NEW CODE (c75f819baf)
#else
  static_assert(false);
#endif
  return 0;
}
