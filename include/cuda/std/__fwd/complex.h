//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _CUDA_STD___FWD_COMPLEX_H
#define _CUDA_STD___FWD_COMPLEX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

// std:: forward declarations

#if _CCCL_HAS_HOST_STD_LIB()
_CCCL_BEGIN_NAMESPACE_STD

template <class>
class complex;

_CCCL_END_NAMESPACE_STD
#endif // _CCCL_HAS_HOST_STD_LIB()

// cuda::std:: forward declarations

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT complex;

// __is_std_complex_v

template <class _Tp>
inline constexpr bool __is_std_complex_v = false;
template <class _Tp>
inline constexpr bool __is_std_complex_v<const _Tp> = __is_std_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_std_complex_v<volatile _Tp> = __is_std_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_std_complex_v<const volatile _Tp> = __is_std_complex_v<_Tp>;
// <<<<<<< OLD CODE from b5d4ca3cf8 (e5037ea8b4) - COMMENTED OUT
// #if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC) // NOTE(HIP/AMD): no host ::std::complex under hipRTC
// template <class _Tp>
// inline constexpr bool __is_std_complex_v<::std::complex<_Tp>> = true;
// #endif // !_CCCL_COMPILER(NVRTC) && !_CCCL_COMPILER_HIPRTC
// =======
#if _CCCL_HOSTED()
template <class _Tp>
inline constexpr bool __is_std_complex_v<::std::complex<_Tp>> = true;
#endif // _CCCL_HOSTED()
// >>>>>>> END NEW CODE (e5037ea8b4)

// __is_cuda_std_complex_v

template <class _Tp>
inline constexpr bool __is_cuda_std_complex_v = false;
template <class _Tp>
inline constexpr bool __is_cuda_std_complex_v<const _Tp> = __is_cuda_std_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_cuda_std_complex_v<volatile _Tp> = __is_cuda_std_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_cuda_std_complex_v<const volatile _Tp> = __is_cuda_std_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_cuda_std_complex_v<complex<_Tp>> = true;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_COMPLEX_H
