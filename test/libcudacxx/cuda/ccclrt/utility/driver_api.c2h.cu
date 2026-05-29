//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/__driver/driver_api.h>

#include <testing.cuh>

// This test is an exception and shouldn't use C2H_CCCLRT_TEST macro
#if !defined(__HIP_PLATFORM_AMD__)
C2H_TEST("Call each driver api", "[utility]")
{
  namespace driver = ::cuda::__driver;
  cudaStream_t stream;
  // Assumes the ctx stack was empty or had one ctx, should be the case unless some other
  // test leaves 2+ ctxs on the stack

  // Pushes the primary context if the stack is empty
  CUDART(cudaStreamCreate(&stream));

  auto ctx = driver::__ctxGetCurrent();
  CCCLRT_REQUIRE(ctx != nullptr);

  // Confirm pop will leave the stack empty
  driver::__ctxPop();
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == nullptr);

  // Confirm we can push multiple times
  driver::__ctxPush(ctx);
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  driver::__ctxPush(ctx);
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  driver::__ctxPop();
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  // Confirm stream ctx match
  auto stream_ctx = driver::__streamGetCtx(stream);
  CCCLRT_REQUIRE(ctx == stream_ctx);

  CUDART(cudaStreamDestroy(stream));

  CCCLRT_REQUIRE(driver::__deviceGet(0) == 0);

  // Confirm we can retain the primary ctx that cudart retained first
  auto primary_ctx = driver::__primaryCtxRetain(0);
  CCCLRT_REQUIRE(ctx == primary_ctx);

  driver::__ctxPop();
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == nullptr);

  CCCLRT_REQUIRE(driver::__isPrimaryCtxActive(0));
  // Confirm we can reset the primary context with double release
  CCCLRT_REQUIRE(driver::__primaryCtxReleaseNoThrow(0) == cudaSuccess);
  CCCLRT_REQUIRE(driver::__primaryCtxReleaseNoThrow(0) == cudaSuccess);

  CCCLRT_REQUIRE(!driver::__isPrimaryCtxActive(0));

  // Confirm cudart can recover
  CUDART(cudaStreamCreate(&stream));
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  CUDART(driver::__streamDestroyNoThrow(stream));
}
#else // ^^^ !__HIP_PLATFORM_AMD__ ^^^ / vvv __HIP_PLATFORM_AMD__ vvv
// NOTE(HIP/AMD): the upstream CUDA-driver-API ctx invariants don't hold
// under HIP's primary-context-only model, and the ctx/primary-ctx
// driver shims have no faithful HIP equivalent (removed from the HIP
// driver_api.h). The HIP subset covers the shims that ARE well-defined:
// device enumeration, version query, and the basic stream lifecycle.
C2H_TEST("Call each driver api (HIP subset)", "[utility]")
{
  namespace driver = ::cuda::__driver;

  // Device query.
  CCCLRT_REQUIRE(driver::__deviceGet(0) == 0);

  // Driver / runtime version is reported.
  CCCLRT_REQUIRE(driver::__getVersion() > 0);

  // Stream lifecycle round-trips.
  cudaStream_t stream{};
  CUDART(cudaStreamCreate(&stream));
  CUDART(driver::__streamDestroyNoThrow(stream));
}
#endif // __HIP_PLATFORM_AMD__
