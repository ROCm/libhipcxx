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

#ifndef _CUDA___COMPLEX_GET_REAL_IMAG_H
#define _CUDA___COMPLEX_GET_REAL_IMAG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/complex.h>
#include <cuda/std/__fwd/complex.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_real(const complex<_Tp>& __c) noexcept
{
  return __c.real();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_imag(const complex<_Tp>& __c) noexcept
{
  return __c.imag();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_real(const ::cuda::std::complex<_Tp>& __c) noexcept
{
  return __c.real();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_imag(const ::cuda::std::complex<_Tp>& __c) noexcept
{
  return __c.imag();
}

#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)

// Unless `--expt-relaxed-constexpr` is specified, obtaining values from std::complex is not constexpr :(
#  if defined(__CUDACC_RELAXED_CONSTEXPR__)
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_real(const ::std::complex<_Tp>& __c) noexcept
{
  return __c.real();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_imag(const ::std::complex<_Tp>& __c) noexcept
{
  return __c.imag();
}
#  else // ^^^ __CUDACC_RELAXED_CONSTEXPR__ ^^^ / vvv !__CUDACC_RELAXED_CONSTEXPR__ vvv
template <class _Tp>
[[nodiscard]] _CCCL_API _Tp __get_real(const ::std::complex<_Tp>& __c) noexcept
{
  return reinterpret_cast<const _Tp(&)[2]>(__c)[0];
}

template <class _Tp>
[[nodiscard]] _CCCL_API _Tp __get_imag(const ::std::complex<_Tp>& __c) noexcept
{
  return reinterpret_cast<const _Tp(&)[2]>(__c)[1];
}
#  endif // ^^^ !__CUDACC_RELAXED_CONSTEXPR__ ^^^
#endif // !_CCCL_COMPILER(NVRTC) && !_CCCL_COMPILER_HIPRTC

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___COMPLEX_GET_REAL_IMAG_H
