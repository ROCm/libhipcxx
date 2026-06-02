// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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

#pragma once

// NOTE(HIP/AMD): use the compiler-provided __HIP_PLATFORM_AMD__ for the
// HIP-vs-CUDA discriminator rather than _CCCL_HIP_COMPILATION() so this
// header doesn't have to pull <cuda/std/detail/__config> just to get the
// macro. __HIP_PLATFORM_AMD__ is defined automatically by hip-clang /
// hipcc on the HIP-AMD path, so no library include is needed; mirrors
// the pattern already used by c2h/bfloat16.cuh and c2h/half.cuh.
#ifndef __HIP_PLATFORM_AMD__
#  include <thrust/detail/config/device_system.h>
#endif // !__HIP_PLATFORM_AMD__

#include <c2h/detail/generators.cuh>

//! @file
//! This file includes a custom Catch2 main function. When CMake is configured to build each test as a separate
//! executable, this header is included into each test. On the other hand, when all the tests are compiled into a single
//! executable, this header is excluded from the tests and included into catch2_runner.cpp

#include <catch2/catch_session.hpp>

// NOTE(HIP/AMD): rocThrust is *optional* under HIP -- the c2h
// CMakeLists does `find_package(rocthrust QUIET CONFIG)` so a HIP build
// without rocThrust still compiles the Catch2-only c2h library. That
// means <thrust/detail/config/device_system.h> may or may not be
// available at parse time on HIP, so we can't rely on
// THRUST_DEVICE_SYSTEM being defined here. Keep the dedicated
// __HIP_PLATFORM_AMD__ arm: HIP always has a (cuda*-aliased-to-hip*)
// runtime regardless of whether rocThrust is wired up, so
// `_C2H_HAS_DEVICE_RUNTIME` is unconditionally true on HIP. The runner
// helper itself routes through cuda* runtime API symbols that are
// shimmed to hip* by <libhipcxx/__amd/cuda_runtime.h>; it does not consume thrust.
#ifdef __HIP_PLATFORM_AMD__
#  define _C2H_HAS_DEVICE_RUNTIME 1
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  define _C2H_HAS_DEVICE_RUNTIME 1
#else
#  define _C2H_HAS_DEVICE_RUNTIME 0
#endif

#ifdef C2H_CONFIG_MAIN
#  if _C2H_HAS_DEVICE_RUNTIME
#    include <c2h/catch2_runner_helper.h>

#    ifndef C2H_EXCLUDE_CATCH2_HELPER_IMPL
#      include "catch2_runner_helper.inl"
#    endif // !C2H_EXCLUDE_CATCH2_HELPER_IMPL
#  endif // _C2H_HAS_DEVICE_RUNTIME

int main(int argc, char* argv[])
{
  Catch::Session session;

#  if _C2H_HAS_DEVICE_RUNTIME
  int device_id{};

  // Build a new parser on top of Catch's
  using namespace Catch::Clara;
  auto cli = session.cli() | Opt(device_id, "device")["-d"]["--device"]("device id to use");
  session.cli(cli);

  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0)
  {
    return returnCode;
  }

  set_device(device_id);
#  endif // _C2H_HAS_DEVICE_RUNTIME
#  if defined(__HIP_PLATFORM_AMD__)
  // NOTE(HIP/AMD): on HIP, cccl.c2h is an INTERFACE library that does not ship the
  // CUB/Thrust generators*.cu, so init_generator/cleanup_generator are unavailable.
  return session.run();
#  else // ^^^ __HIP_PLATFORM_AMD__ ^^^ / vvv !__HIP_PLATFORM_AMD__ vvv
  c2h::detail::init_generator();
  const auto ret = session.run();
  c2h::detail::cleanup_generator();
  return ret;
#  endif // !__HIP_PLATFORM_AMD__
}
#endif // C2H_CONFIG_MAIN
