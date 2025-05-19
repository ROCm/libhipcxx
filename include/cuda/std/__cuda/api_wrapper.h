//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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

#ifndef _CUDA__STD__CUDA_API_WRAPPER_H
#define _CUDA__STD__CUDA_API_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// <<<<<<< OLD CODE from 40c5b56c00 (f69d5ba4be) - COMMENTED OUT
// #if _CCCL_CUDA_COMPILER(CLANG) && !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC_RTC__)
// #  include <cuda_runtime_api.h>
// #endif // _CCCL_CUDA_COMPILER(CLANG)
//
// =======
// >>>>>>> END NEW CODE (f69d5ba4be)
#include <cuda/std/__exception/cuda_error.h>

// <<<<<<< OLD CODE from 9de450753d (ecc4210681) - COMMENTED OUT
// // <<<<<<< OLD CODE from 6a706b2d84 (178551348f) - COMMENTED OUT
// // #if _CCCL_HAS_CUDA_COMPILER()
// // #  define _CCCL_TRY_CUDA_API(_NAME, _MSG, ...)           \
// //     {                                                    \
// //       const ::cudaError_t __status = _NAME(__VA_ARGS__); \
// //       switch (__status)                                  \
// //       {                                                  \
// //         case ::cudaSuccess:                              \
// //           break;                                         \
// //         default:                                         \
// //           ::cudaGetLastError();                          \
// //           ::hip::__throw_cuda_error(__status, _MSG);    \
// //       }                                                  \
// //     }
// //
// // #  define _CCCL_ASSERT_CUDA_API(_NAME, _MSG, ...)                         \
// //     {                                                                     \
// //       [[maybe_unused]] const ::cudaError_t __status = _NAME(__VA_ARGS__); \
// //       _CCCL_ASSERT(__status == cudaSuccess, _MSG);                        \
// //     }
// // #else // ^^^ _CCCL_HAS_CUDA_COMPILER() ^^^ / vvv !_CCCL_HAS_CUDA_COMPILER() vvv
// // #  define _CCCL_TRY_CUDA_API(_NAME, _MSG, ...)
// // #  define _CCCL_ASSERT_CUDA_API(_NAME, _MSG, ...)
// // #endif // !_CCCL_HAS_CUDA_COMPILER()
// // =======
// #define _CCCL_TRY_CUDA_API(_NAME, _MSG, ...)           \
//   {                                                    \
//     const ::cudaError_t __status = _NAME(__VA_ARGS__); \
//     switch (__status)                                  \
//     {                                                  \
//       case ::cudaSuccess:                              \
//         break;                                         \
//       default:                                         \
//         ::cudaGetLastError();                          \
//         ::cuda::__throw_cuda_error(__status, _MSG);    \
//     }                                                  \
// =======
#define _CCCL_TRY_CUDA_API(_NAME, _MSG, ...)                \
  {                                                         \
    const ::cudaError_t __status = _NAME(__VA_ARGS__);      \
    switch (__status)                                       \
    {                                                       \
      case ::cudaSuccess:                                   \
        break;                                              \
      default:                                              \
        ::cudaGetLastError();                               \
        ::cuda::__throw_cuda_error(__status, _MSG, #_NAME); \
    }                                                       \
// >>>>>>> END NEW CODE (ecc4210681)
  }

#define _CCCL_ASSERT_CUDA_API(_NAME, _MSG, ...)                         \
  {                                                                     \
    [[maybe_unused]] const ::cudaError_t __status = _NAME(__VA_ARGS__); \
    _CCCL_ASSERT(__status == cudaSuccess, _MSG);                        \
  }
// >>>>>>> END NEW CODE (178551348f)

#define _CCCL_LOG_CUDA_API(_NAME, _MSG, ...)                                       \
  [&]() {                                                                          \
    const ::cudaError_t __status = _NAME(__VA_ARGS__);                             \
    if (__status != ::cudaSuccess)                                                 \
    {                                                                              \
      ::cuda::__detail::__msg_storage __msg_buffer;                                \
      ::cuda::__detail::__format_cuda_error(__msg_buffer, __status, _MSG, #_NAME); \
      ::fprintf(stderr, "%s\n", __msg_buffer.__buffer);                            \
      ::fflush(stderr);                                                            \
    }                                                                              \
    return __status;                                                               \
  }()

#endif //_CUDA__STD__CUDA_API_WRAPPER_H
