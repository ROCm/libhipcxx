//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <<<<<<< OLD CODE from 83c81aee48 (c1adecddac) - COMMENTED OUT
// // MIT License
// //
// // Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// XFAIL: enable-tile
// nvbug6067464: error: Internal Compiler Error (tile codegen): "call to unknown tile builtin function!
// >>>>>>> END NEW CODE (c1adecddac)

#include <cuda/mdspan>

#include "test_macros.h"

TEST_DEVICE_FUNC void basic_mdspan_access_test()
{
  using ext_t = cuda::std::extents<int, 4>;
  __shared__ int smem[4];
  [[maybe_unused]] cuda::shared_memory_mdspan<int, ext_t> md{smem, ext_t{}};
  unused(md[0]);
  // NOTE(HIP/AMD): "l" is an NVPTX 64-bit register constraint; on amdgcn use a
  // vector-register ("v") constraint to keep the shared-memory pointer live.
#if defined(__HIP_PLATFORM_AMD__)
  asm volatile("" : : "v"((size_t) smem) : "memory");
#else
  asm volatile("" : : "l"((size_t) smem) : "memory");
#endif
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (basic_mdspan_access_test();))
  return 0;
}
