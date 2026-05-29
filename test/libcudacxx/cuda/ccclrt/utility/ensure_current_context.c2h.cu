//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/devices>

#include <testing.cuh>

namespace driver = cuda::__driver;

void recursive_check_device_setter(int id)
{
  int cudart_id;
  cuda::__ensure_current_context setter(cuda::device_ref{id});
  // Driver-API ctx-stack invariants only apply on CUDA; the
  // cudaGetDevice() check below still exercises the observable
  // effect of __ensure_current_context (hipSetDevice) on HIP.
#if !_CCCL_HIP_COMPILATION()
  CCCLRT_REQUIRE(test::count_driver_stack() == cuda::devices.size() - id);
  auto ctx = driver::__ctxGetCurrent();
#endif // !_CCCL_HIP_COMPILATION()
  CUDART(cudaGetDevice(&cudart_id));
  CCCLRT_REQUIRE(cudart_id == id);

  if (id != 0)
  {
    recursive_check_device_setter(id - 1);

#if !_CCCL_HIP_COMPILATION()
    // Post-unwind assertions only apply in the CUDA driver-API stack
    // model. HIP has no ctx stack; the pre-recursion cudaGetDevice
    // check above still holds and keeps running.
    CCCLRT_REQUIRE(test::count_driver_stack() == cuda::devices.size() - id);
    CCCLRT_REQUIRE(ctx == driver::__ctxGetCurrent());
    CUDART(cudaGetDevice(&cudart_id));
    CCCLRT_REQUIRE(cudart_id == id);
#else
    (void) cudart_id;
#endif // !_CCCL_HIP_COMPILATION()
  }
}

C2H_TEST("ensure current context", "[device]")
{
  test::empty_driver_stack();
  // If possible use something different than CUDART default 0
  int target_device = static_cast<int>(cuda::devices.size() - 1);

  SECTION("context setter")
  {
    recursive_check_device_setter(target_device);

#if !_CCCL_HIP_COMPILATION()
    CCCLRT_REQUIRE(test::count_driver_stack() == 0);
#endif // !_CCCL_HIP_COMPILATION()
  }
}
