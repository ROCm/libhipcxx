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

#ifndef _CUDA_STD___CHRONO_SYSTEM_CLOCK_H
#define _CUDA_STD___CHRONO_SYSTEM_CLOCK_H

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

#if !_CCCL_COMPILER(NVRTC)
#  include <chrono>
#endif // !_CCCL_COMPILER(NVRTC)

// NOTE(HIP/AMD): C++20 UNIX-timestamp opt-in workaround for system_clock on
// AMD GPUs. The extension header declares cuda::std::chrono::hip_gpu_ext::
// __unix_sysclock0_host_ticks / __offset_devclock0 plus the macro
// LIBCUDACXX_HIP_DEFINE_SYSCLOCK_VARS and the host-side
// initialize_amdgpu_sysclock_on_{current_,}device() helpers used by
// system_clock::now() below. See the header itself for the full opt-in
// protocol.
#if _CCCL_HIP_COMPILATION() && !defined(_CCCL_COMPILER_HIPRTC)
#  if _CCCL_STD_VER > 2017 && defined(_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP)
#    include <libhipcxx/__amd/hip_chrono_extension.h>
#  endif // _CCCL_STD_VER > 2017 && _LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP
#endif // _CCCL_HIP_COMPILATION() && !_CCCL_COMPILER_HIPRTC

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class _CCCL_TYPE_VISIBILITY_DEFAULT system_clock
{
public:
  using duration                  = ::cuda::std::chrono::nanoseconds;
  using rep                       = duration::rep;
  using period                    = duration::period;
  using time_point                = ::cuda::std::chrono::time_point<system_clock>;
  static constexpr bool is_steady = false;

  [[nodiscard]] _CCCL_API inline static time_point now() noexcept
  {
#if _CCCL_CUDA_COMPILATION()
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (return time_point(duration_cast<duration>(nanoseconds(
        ::std::chrono::duration_cast<::std::chrono::nanoseconds>(::std::chrono::system_clock::now().time_since_epoch())
          .count())));),
      (return time_point(duration_cast<duration>(nanoseconds(::cuda::ptx::get_sreg_globaltimer())));))
#else // ^^^ _CCCL_CUDA_COMPILATION() ^^^ / vvv !_CCCL_CUDA_COMPILATION() vvv
    // NOTE(HIP/AMD): AMD GPUs don't expose a UNIX-timestamp counter,
    // so emulate cuda::ptx::get_sreg_globaltimer() via two paths:
    //  - default: wall_clock64() (TSC cycles) scaled by the arch-
    //    dependent _LIBCUDACXX_HIP_TSC_CLOCKRATE from
    //    <libhipcxx/__amd/hip_tsc_clockrate.h>. NOT a UNIX timestamp.
    //  - opt-in (-D_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP): use the
    //    host-initialised offsets in cuda::std::chrono::hip_gpu_ext
    //    so the device-side time_point IS a UNIX timestamp. See
    //    <libhipcxx/__amd/hip_chrono_extension.h> for the protocol.
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (return time_point(duration_cast<duration>(nanoseconds(
        ::std::chrono::duration_cast<::std::chrono::nanoseconds>(::std::chrono::system_clock::now().time_since_epoch())
          .count())));),
#  if _CCCL_STD_VER > 2017 && defined(_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP)
      // FIXME(HIP/AMD): UNIX-timestamp workaround. Requires user-side
      // initialisation -- see chrono_hip_extension.h for the protocol.
      (if (!(::cuda::std::chrono::hip_gpu_ext::__unix_sysclock0_host_ticks >= 0))
       {
         // FIXME(HIP/AMD): now() is noexcept, so we cannot throw here.
         printf("ERROR: Using sysclock on AMD GPUs requires a prior initialization call on the host side "
                "(cuda::std::chrono::hip_gpu_ext::initialize_amdgpu_sysclock_on_current_device()). "
                "The returned time point will not be a UNIX timestamp.\n");
       }
       assert(::cuda::std::chrono::hip_gpu_ext::__unix_sysclock0_host_ticks >= 0);
       // Convert host ticks to device ticks via the arch TSC rate, then add
       // the device-side delta since the host-side initialisation moment.
       const long long __unix_sysclock0_device_ticks =
         ::cuda::std::chrono::hip_gpu_ext::__unix_sysclock0_host_ticks
         / _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE;
       const long long __time =
         __unix_sysclock0_device_ticks
         + (wall_clock64() - ::cuda::std::chrono::hip_gpu_ext::__offset_devclock0);
       return time_point(duration_cast<duration>(
         ::cuda::std::chrono::duration<long long, ratio<1, _LIBCUDACXX_HIP_TSC_CLOCKRATE>>(__time)));))
#  else // ^^^ _LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP ^^^ / vvv default vvv
      (const long long __cycles = wall_clock64();
       return time_point(duration_cast<duration>(
         ::cuda::std::chrono::duration<long long, ratio<1, _LIBCUDACXX_HIP_TSC_CLOCKRATE>>(__cycles)));))
#  endif // !_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP
#endif // !_CCCL_CUDA_COMPILATION()
  }

  [[nodiscard]] _CCCL_API inline static time_t to_time_t(const time_point& __t) noexcept
  {
    return time_t(::cuda::std::chrono::duration_cast<seconds>(__t.time_since_epoch()).count());
  }

  [[nodiscard]] _CCCL_API inline static time_point from_time_t(time_t __t) noexcept
  {
    return time_point(::cuda::std::chrono::seconds(__t));
  }
};

template <class _Duration>
using sys_time    = time_point<system_clock, _Duration>;
using sys_seconds = sys_time<seconds>;
using sys_days    = sys_time<days>;
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_SYSTEM_CLOCK_H
