// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#ifndef _CUDA_STD___PSTL_CUDA_HIPCUB_H
#define _CUDA_STD___PSTL_CUDA_HIPCUB_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// NOTE(HIP/AMD): the libcudacxx PSTL "cuda" backend (cuda/std/__pstl/cuda/*.h)
// is written against NVIDIA CUB -- it calls ::cub::DeviceReduce / ::cub::DeviceFor
// and relies on cub/device/*.cuh headers. On HIP we reuse that backend code
// VERBATIM instead of forking it, using the same trick the c2h test headers use
// (c2h/include/c2h/half.cuh: `namespace cub = hipcub;`):
//   * pull hipCUB's device primitives (these provide hipcub::DeviceReduce /
//     hipcub::DeviceFor with CUB-compatible signatures), and
//   * alias `namespace cub = ::hipcub` so the unmodified `::cub::Device*` call
//     sites resolve to hipCUB.
// The cuda* runtime symbols the backend uses (cudaStream_t, cudaError_t,
// cudaStreamPerThread, cudaMemcpyAsync, cudaMemcpyDefault, cudaErrorMemoryAllocation)
// are already HIP-aliased globally via <cuda/std/detail/__config> ->
// <libhipcxx/__amd/cuda_runtime.h>, and _CCCL_TRY_CUDA_API works on HIP, so the
// backend bodies need no further changes for these primitives.
//
// This currently covers the algorithms whose backend maps to DeviceReduce /
// DeviceFor (reduce, count, count_if, for_each, for_each_n). The transform /
// generate backends use CUB-internal APIs (cub::detail::transform::dispatch,
// cub::DeviceTransform::Generate) that hipCUB does not expose; they are retargeted
// to the public hipcub::DeviceTransform::Transform separately and remain gated off
// on HIP for now.
#if _CCCL_HIP_COMPILATION()

#  include <hipcub/device/device_for.hpp>
#  include <hipcub/device/device_reduce.hpp>

// CUB_NS_QUALIFIER is a CUB-only macro (cub/util_namespace.cuh) that hipCUB does
// not define. Point it at hipcub so any CUB_NS_QUALIFIER:: use resolves on HIP.
#  ifndef CUB_NS_QUALIFIER
#    define CUB_NS_QUALIFIER ::hipcub
#  endif // CUB_NS_QUALIFIER

namespace cub = ::hipcub;

#endif // _CCCL_HIP_COMPILATION()

#endif // _CUDA_STD___PSTL_CUDA_HIPCUB_H
