// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

#ifndef _CUDA_STD___CHRONO_STEADY_CLOCK_H
#define _CUDA_STD___CHRONO_STEADY_CLOCK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_MONOTONIC_CLOCK()

#  include <cuda/std/__chrono/duration.h>
#  include <cuda/std/__chrono/time_point.h>

// <<<<<<< OLD CODE from b5d4ca3cf8 (e5037ea8b4) - COMMENTED OUT
// #  if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC) // NOTE(HIP/AMD): no host <chrono> under hipRTC
// #    include <chrono>
// #  endif // !_CCCL_COMPILER(NVRTC) && !_CCCL_COMPILER_HIPRTC
// =======
#  if _CCCL_HOSTED()
#    include <chrono>
#  endif // _CCCL_HOSTED()
// >>>>>>> END NEW CODE (e5037ea8b4)

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class _CCCL_TYPE_VISIBILITY_DEFAULT steady_clock
{
public:
  using duration                  = nanoseconds;
  using rep                       = duration::rep;
  using period                    = duration::period;
  using time_point                = ::cuda::std::chrono::time_point<steady_clock, duration>;
  static constexpr bool is_steady = true;

  [[nodiscard]] _CCCL_API static time_point now() noexcept;
};
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX_HAS_MONOTONIC_CLOCK()

#endif // _CUDA_STD___CHRONO_STEADY_CLOCK_H
