//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <<<<<<< OLD CODE from 3398a4cc61 (4eeba3d3c2) - COMMENTED OUT
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
//
// // UNSUPPORTED: nvrtc, hiprtc
// =======
// UNSUPPORTED: enable-tile

// UNSUPPORTED: nvrtc
// >>>>>>> END NEW CODE (4eeba3d3c2)

#include <cuda/mdspan>

#include "test_macros.h"

using ext_t = cuda::std::extents<int, 4>;

bool host_accessor_test()
{
  int array[] = {1, 2, 3, 4};
  int* h_ptr;
  assert(cudaMallocHost(&h_ptr, 4) == cudaSuccess);
  cuda::host_mdspan<int, ext_t> h_md{array, ext_t{}};
  cuda::host_mdspan<int, ext_t> h_md2{h_ptr, ext_t{}};
  unused(h_md);
  unused(h_md2);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(host_accessor_test());))
  return 0;
}
