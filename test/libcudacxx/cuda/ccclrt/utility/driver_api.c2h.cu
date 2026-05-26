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
// NOTE(HIP/AMD): the upstream CUDA-driver-API invariants don't hold
// under HIP's primary-context-only model (no real ctx-stack pop,
// fresh hipCtx_t handle per __ctxGetCurrent() query, no stream-ctx
// API, sticky primary-ctx 'active' flag). The HIP subset below
// covers what IS well-defined regardless of ctx semantics: device
// enumeration, version query, primary-ctx retain/release refcount,
// and the basic stream lifecycle.
C2H_TEST("Call each driver api (HIP subset)", "[utility]")
{
  namespace driver = ::cuda::__driver;

  // Device query.
  CCCLRT_REQUIRE(driver::__deviceGet(0) == 0);

  // Driver / runtime version is reported.
  CCCLRT_REQUIRE(driver::__getVersion() > 0);

  // Stream lifecycle.
  cudaStream_t stream{};
  CUDART(cudaStreamCreate(&stream));

  // After a stream is created the runtime has touched the primary
  // context, so __ctxGetCurrent reports a non-null wrapper handle.
  auto ctx = driver::__ctxGetCurrent();
  CCCLRT_REQUIRE(ctx != nullptr);

  // We can retain + release the primary context (refcount).
  auto primary_ctx = driver::__primaryCtxRetain(0);
  CCCLRT_REQUIRE(primary_ctx != nullptr);
  CCCLRT_REQUIRE(driver::__primaryCtxReleaseNoThrow(0) == cudaSuccess);

  // Force-activate the primary context. ROCm 7.2 does not flip the
  // 'active' flag from hipStreamCreate or hipDevicePrimaryCtxRetain
  // alone; only an actual GPU runtime operation does. Use a tiny
  // hipMalloc/hipFree round-trip.
  {
    void* __dummy = nullptr;
    if (cudaMalloc(&__dummy, 1) == cudaSuccess)
    {
      (void) cudaFree(__dummy);
    }
  }
  CCCLRT_REQUIRE(driver::__isPrimaryCtxActive(driver::__deviceGet(0)));

  // Cleanup: the stream lifecycle round-trips.
  CUDART(driver::__streamDestroyNoThrow(stream));
}
#endif // __HIP_PLATFORM_AMD__
