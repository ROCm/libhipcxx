//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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

#ifndef _LIBCUDACXX___TUPLE_STRUCTURED_BINDINGS_H
#define _LIBCUDACXX___TUPLE_STRUCTURED_BINDINGS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wmismatched-tags")

#if !_CCCL_COMPILER(NVRTC)
// Fetch utility to get primary template for ::std::tuple_size necessary for the specialization of
// ::std::tuple_size<cuda::std::tuple> to enable structured bindings.
// See https://github.com/NVIDIA/libcudacxx/issues/316
// NOTE(HIP/AMD): Also needed under HIPRTC since libstdc++ provides std::tuple_size
// and our partial specializations need its primary template to be declared.
#  include <utility>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/subrange.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/integral_constant.h>

// This is a workaround for the fact that structured bindings require that the specializations of
// `tuple_size` and `tuple_element` reside in namespace std (https://eel.is/c++draft/dcl.struct.bind#4).
// See https://github.com/NVIDIA/libcudacxx/issues/316 for a short discussion

#include <cuda/std/__cccl/prologue.h>

namespace std
{
// NOTE(HIP/AMD): When system <utility> isn't available (NVRTC), provide forward
// declarations of tuple_size/tuple_element using variadic template parameters.
// Under HIPRTC, we include <utility> to get the primary template from libstdc++.
#if _CCCL_COMPILER(NVRTC)
template <class... _Tp>
struct tuple_size;

template <size_t _Ip, class... _Tp>
struct tuple_element;
#endif // _CCCL_COMPILER(NVRTC)

// NOTE(HIP/AMD): Array specializations work with libstdc++'s std::tuple_size
// since they're proper partial specializations.
template <class _Tp, size_t _Size>
struct tuple_size<::cuda::std::array<_Tp, _Size>> : ::cuda::std::tuple_size<::cuda::std::array<_Tp, _Size>>
{};

template <class _Tp, size_t _Size>
struct tuple_size<const ::cuda::std::array<_Tp, _Size>> : ::cuda::std::tuple_size<::cuda::std::array<_Tp, _Size>>
{};

template <class _Tp, size_t _Size>
struct tuple_size<volatile ::cuda::std::array<_Tp, _Size>> : ::cuda::std::tuple_size<::cuda::std::array<_Tp, _Size>>
{};

template <class _Tp, size_t _Size>
struct tuple_size<const volatile ::cuda::std::array<_Tp, _Size>>
    : ::cuda::std::tuple_size<::cuda::std::array<_Tp, _Size>>
{};

template <size_t _Ip, class _Tp, size_t _Size>
struct tuple_element<_Ip, ::cuda::std::array<_Tp, _Size>>
    : ::cuda::std::tuple_element<_Ip, ::cuda::std::array<_Tp, _Size>>
{};

template <size_t _Ip, class _Tp, size_t _Size>
struct tuple_element<_Ip, const ::cuda::std::array<_Tp, _Size>>
    : ::cuda::std::tuple_element<_Ip, const ::cuda::std::array<_Tp, _Size>>
{};

template <size_t _Ip, class _Tp, size_t _Size>
struct tuple_element<_Ip, volatile ::cuda::std::array<_Tp, _Size>>
    : ::cuda::std::tuple_element<_Ip, volatile ::cuda::std::array<_Tp, _Size>>
{};

template <size_t _Ip, class _Tp, size_t _Size>
struct tuple_element<_Ip, const volatile ::cuda::std::array<_Tp, _Size>>
    : ::cuda::std::tuple_element<_Ip, const volatile ::cuda::std::array<_Tp, _Size>>
{};

// NOTE(HIP/AMD): These partial specializations work with both libstdc++ and our
// tuple_size/tuple_element since they are single-template-parameter specializations.
template <class _Tp>
struct tuple_size<::cuda::std::complex<_Tp>> : ::cuda::std::tuple_size<::cuda::std::complex<_Tp>>
{};

template <size_t _Ip, class _Tp>
struct tuple_element<_Ip, ::cuda::std::complex<_Tp>> : ::cuda::std::tuple_element<_Ip, ::cuda::std::complex<_Tp>>
{};

template <class _Tp, class _Up>
struct tuple_size<::cuda::std::pair<_Tp, _Up>> : ::cuda::std::tuple_size<::cuda::std::pair<_Tp, _Up>>
{};

template <class _Tp, class _Up>
struct tuple_size<const ::cuda::std::pair<_Tp, _Up>> : ::cuda::std::tuple_size<::cuda::std::pair<_Tp, _Up>>
{};

template <class _Tp, class _Up>
struct tuple_size<volatile ::cuda::std::pair<_Tp, _Up>> : ::cuda::std::tuple_size<::cuda::std::pair<_Tp, _Up>>
{};

template <class _Tp, class _Up>
struct tuple_size<const volatile ::cuda::std::pair<_Tp, _Up>> : ::cuda::std::tuple_size<::cuda::std::pair<_Tp, _Up>>
{};

template <size_t _Ip, class _Tp, class _Up>
struct tuple_element<_Ip, ::cuda::std::pair<_Tp, _Up>> : ::cuda::std::tuple_element<_Ip, ::cuda::std::pair<_Tp, _Up>>
{};

template <size_t _Ip, class _Tp, class _Up>
struct tuple_element<_Ip, const ::cuda::std::pair<_Tp, _Up>>
    : ::cuda::std::tuple_element<_Ip, const ::cuda::std::pair<_Tp, _Up>>
{};

template <size_t _Ip, class _Tp, class _Up>
struct tuple_element<_Ip, volatile ::cuda::std::pair<_Tp, _Up>>
    : ::cuda::std::tuple_element<_Ip, volatile ::cuda::std::pair<_Tp, _Up>>
{};

template <size_t _Ip, class _Tp, class _Up>
struct tuple_element<_Ip, const volatile ::cuda::std::pair<_Tp, _Up>>
    : ::cuda::std::tuple_element<_Ip, const volatile ::cuda::std::pair<_Tp, _Up>>
{};

// NOTE(HIP/AMD): Partial specializations for tuple/subrange work with libstdc++
// and libc++ since they're proper partial specializations of the standard's
// tuple_size/tuple_element template.
template <class... _Tp>
struct tuple_size<::cuda::std::tuple<_Tp...>> : ::cuda::std::tuple_size<::cuda::std::tuple<_Tp...>>
{};

template <class... _Tp>
struct tuple_size<const ::cuda::std::tuple<_Tp...>> : ::cuda::std::tuple_size<::cuda::std::tuple<_Tp...>>
{};

template <class... _Tp>
struct tuple_size<volatile ::cuda::std::tuple<_Tp...>> : ::cuda::std::tuple_size<::cuda::std::tuple<_Tp...>>
{};

template <class... _Tp>
struct tuple_size<const volatile ::cuda::std::tuple<_Tp...>> : ::cuda::std::tuple_size<::cuda::std::tuple<_Tp...>>
{};

template <size_t _Ip, class... _Tp>
struct tuple_element<_Ip, ::cuda::std::tuple<_Tp...>> : ::cuda::std::tuple_element<_Ip, ::cuda::std::tuple<_Tp...>>
{};

template <size_t _Ip, class... _Tp>
struct tuple_element<_Ip, const ::cuda::std::tuple<_Tp...>>
    : ::cuda::std::tuple_element<_Ip, const ::cuda::std::tuple<_Tp...>>
{};

template <size_t _Ip, class... _Tp>
struct tuple_element<_Ip, volatile ::cuda::std::tuple<_Tp...>>
    : ::cuda::std::tuple_element<_Ip, volatile ::cuda::std::tuple<_Tp...>>
{};

template <size_t _Ip, class... _Tp>
struct tuple_element<_Ip, const volatile ::cuda::std::tuple<_Tp...>>
    : ::cuda::std::tuple_element<_Ip, const volatile ::cuda::std::tuple<_Tp...>>
{};

template <class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_size<::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
    : ::cuda::std::tuple_size<::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{};

template <class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_size<const ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
    : ::cuda::std::tuple_size<::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{};

template <class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_size<volatile ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
    : ::cuda::std::tuple_size<::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{};

template <class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_size<const volatile ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
    : ::cuda::std::tuple_size<::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{};

template <size_t _Idx, class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_element<_Idx, ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
    : ::cuda::std::tuple_element<_Idx, ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{};

template <size_t _Idx, class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_element<_Idx, const ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
    : ::cuda::std::tuple_element<_Idx, const ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{};

template <size_t _Idx, class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_element<_Idx, volatile ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
    : ::cuda::std::tuple_element<_Idx, volatile ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{};

template <size_t _Idx, class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_element<_Idx, const volatile ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
    : ::cuda::std::tuple_element<_Idx, const volatile ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{};
} // namespace std

#include <cuda/std/__cccl/epilogue.h>

_CCCL_DIAG_POP

#endif // _LIBCUDACXX___TUPLE_STRUCTURED_BINDINGS_H
