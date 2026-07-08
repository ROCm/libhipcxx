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

#ifndef _CUDA_STD___CHRONO_FILE_CLOCK_H
#define _CUDA_STD___CHRONO_FILE_CLOCK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/time_point.h>
#include <cuda/std/ctime>

#if _CCCL_HOSTED() // NOTE(HIP/AMD): no host <chrono> under hipRTC
#  include <chrono>
#endif // _CCCL_HOSTED()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_FILESYSTEM
struct _FilesystemClock;
_CCCL_END_NAMESPACE_FILESYSTEM

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
// [time.clock.file], type file_clock
using file_clock = ::cuda::std::__fs::filesystem::_FilesystemClock;

template <class _Duration>
using file_time = time_point<file_clock, _Duration>;
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#ifndef __cuda_std__

_CCCL_BEGIN_NAMESPACE_FILESYSTEM
struct _FilesystemClock
{
#  if _CCCL_HAS_INT128()
  using rep    = __int128_t;
  using period = nano;
#  else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
  using rep    = long long;
  using period = nano;
#  endif // !_CCCL_HAS_INT128()

  using duration   = chrono::duration<rep, period>;
  using time_point = chrono::time_point<_FilesystemClock>;

  _CCCL_VISIBILITY_DEFAULT static constexpr const bool is_steady = false;

  [[nodiscard]] _CCCL_API inline static time_point now() noexcept;

  [[nodiscard]] _CCCL_API inline static time_t to_time_t(const time_point& __t) noexcept
  {
    using __secs = chrono::duration<rep>;
    return time_t(chrono::duration_cast<__secs>(__t.time_since_epoch()).count());
  }

  [[nodiscard]] _CCCL_API inline static time_point from_time_t(time_t __t) noexcept
  {
    using __secs = chrono::duration<rep>;
    return time_point(__secs(__t));
  }
};
_CCCL_END_NAMESPACE_FILESYSTEM
#endif // __cuda_std__

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_FILE_CLOCK_H
