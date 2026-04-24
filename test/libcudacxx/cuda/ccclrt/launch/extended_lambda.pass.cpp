//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <<<<<<< OLD CODE from a68250b35b (7864e1c9a1) - COMMENTED OUT
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
// error: indirect call is unsupported in tile code
// >>>>>>> END NEW CODE (7864e1c9a1)

// ADDITIONAL_COMPILE_FLAGS: --extended-lambda
// UNSUPPORTED: nvrtc, hiprtc

#include <cuda/devices>
#include <cuda/launch>
#include <cuda/stream>

#include "../common/utility.cuh"
#include "test_macros.h"

void test_extended_lambda()
{
  cuda::stream stream{cuda::devices[0]};
  test::pinned<int> i(0);
  auto config           = cuda::block_dims<32>() & cuda::grid_dims<1>();
  auto assign_42_lambda = [] TEST_DEVICE_FUNC(int* pi) {
    *pi = 42;
  };
  cuda::launch(stream, config, assign_42_lambda, i.get());
  stream.sync();
  assert(*i == 42);

  auto assign_1337_lambda = [] TEST_DEVICE_FUNC(auto config, int* pi) {
    static_assert(cuda::gpu_thread.count(cuda::block, config) == 32);
    static_assert(cuda::block.count(cuda::grid, config) == 1);
    *pi = 1337;
  };
  cuda::launch(stream, config, assign_1337_lambda, config, i.get());
  stream.sync();
  assert(*i == 1337);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test_extended_lambda();)
  return 0;
}
