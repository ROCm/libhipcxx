// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef _LIBCUDACXX___CUDA_CHRONO_H
#define _LIBCUDACXX___CUDA_CHRONO_H

#ifndef __cuda_std__
#error "<__cuda/chrono> should only be included in from <cuda/std/chrono>"
#endif // __cuda_std__

#include <nv/target>

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace chrono {

#if defined(__HIP__) || defined(__HIPCC_RTC__)
#if _LIBCUDACXX_STD_VER>17 && defined(_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP)
// Workaround for system_clock on AMD GPUs for c++20, please see documentation in the below header file
#include "../support/hip/chrono_hip_extension.h"
#endif
#endif

inline _LIBCUDACXX_INLINE_VISIBILITY
system_clock::time_point system_clock::now() _NOEXCEPT
{
// FIXME(HIP): simplify logic and use NV_DISPATCH_TARGET macros
#ifdef __CUDA_ARCH__
    uint64_t __time;
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(__time)::);
    return time_point(duration_cast<duration>(nanoseconds(__time)));
#elif defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC_RTC__)
#if _LIBCUDACXX_STD_VER>17 
#if defined(_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP)
    if(!(hip_gpu_ext::__unix_sysclock0_host_ticks>=0)) {
        // FIXME(HIP): As this function needs to be NOEXCEPT, we can't throw an exception at this point.
        printf("ERROR: Using sysclock on AMD GPUs requires a prior initialization call on the host side (cuda::std::chrono::hip_gpu_ext::initialize_amdgpu_sysclock_on_current_device())."
             "The returned time point will not be a UNIX timestamp.\n");
    }
    // FIXME(HIP): This compilation path uses a workaround to make a UNIX timestamp counter available on the device.
    // see header "detail/libcxx/include/support/hip/chrono_hip_extension.h" for more details. 
    assert(hip_gpu_ext::__unix_sysclock0_host_ticks>=0);

    // convert host ticks to device ticks
    long long __unix_sysclock0_device_ticks = hip_gpu_ext::__unix_sysclock0_host_ticks / _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE; 

    long long __time = __unix_sysclock0_device_ticks 
                    + (wall_clock64()-hip_gpu_ext::__offset_devclock0);
    return time_point(duration_cast<duration>(chrono::duration<long long, ratio<1,_LIBCUDACXX_HIP_TSC_CLOCKRATE>>(__time)));
#else
    printf("WARNING: A C++20 standard-conform system_clock is currently only supported with an experimental workaround that can be "
           "activated with -D_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP compile flag (see detail/libcxx/include/support/hip/chrono_hip_extension.h)."
           "The returned time point will not be a UNIX timestamp.\n");
    // default HIP implementation C++20 without UNIX timestamp workaround
    // FIXME(HIP): Enable UNIX timestamps on the device without any workaround.
    long long __time = wall_clock64();
    return time_point(duration_cast<duration>(chrono::duration<long long, ratio<1,_LIBCUDACXX_HIP_TSC_CLOCKRATE>>(__time)));
#endif /*_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP*/
#else
    // FIXME(HIP): The timestamp on AMD devices will not be a UNIX timestamp. It is therefore different from the one on the host.
    // default HIP implementation
    long long __time = wall_clock64();
    return time_point(duration_cast<duration>(chrono::duration<long long, ratio<1,_LIBCUDACXX_HIP_TSC_CLOCKRATE>>(__time)));
#endif /*_LIBCUDACXX_STD_VER>17*/
#else
    return time_point(duration_cast<duration>(nanoseconds(
            ::std::chrono::duration_cast<::std::chrono::nanoseconds>(
                ::std::chrono::system_clock::now().time_since_epoch()
            ).count()
           )));
#endif
}

inline _LIBCUDACXX_INLINE_VISIBILITY
time_t system_clock::to_time_t(const system_clock::time_point& __t) _NOEXCEPT
{
    return time_t(duration_cast<seconds>(__t.time_since_epoch()).count());
}

inline _LIBCUDACXX_INLINE_VISIBILITY
system_clock::time_point system_clock::from_time_t(time_t __t) _NOEXCEPT
{
    return time_point(seconds(__t));;
}
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CUDA_CHRONO_H
