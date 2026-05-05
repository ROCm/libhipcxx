//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _LIBCUDACXX___INTERNAL_NAMESPACES_H
#define _LIBCUDACXX___INTERNAL_NAMESPACES_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// During the header testing, we want to check if the code is wrapped by the prologue/epilogue
#if defined(_CCCL_HEADER_TEST)
#  define _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() \
    static_assert(_CCCL_PROLOGUE_INCLUDED(), "missing #include <cuda/std/__cccl/prologue.h>");
#else // ^^^ defined(_CCCL_HEADER_TEST) ^^^ / vvv !defined(_CCCL_HEADER_TEST) vvv
#  define _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#endif // ^^^ !defined(_CCCL_HEADER_TEST) ^^^

#define _LIBCUDACXX_CONCAT1(_LIBCUDACXX_X, _LIBCUDACXX_Y) _LIBCUDACXX_X##_LIBCUDACXX_Y
#define _LIBCUDACXX_CONCAT(_LIBCUDACXX_X, _LIBCUDACXX_Y)  _LIBCUDACXX_CONCAT1(_LIBCUDACXX_X, _LIBCUDACXX_Y)

#ifndef _LIBCUDACXX_ABI_NAMESPACE
#  define _LIBCUDACXX_ABI_NAMESPACE _LIBCUDACXX_CONCAT(__, _LIBCUDACXX_CUDA_ABI_VERSION)
#endif // _LIBCUDACXX_ABI_NAMESPACE

// clang-format off

// Standard namespaces with or without versioning
#  define _LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std {
#  define _LIBCUDACXX_END_NAMESPACE_STD_NOVERSION } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_STD _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_STD } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;

// cuda specific namespaces
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::mr { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_MR } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::device { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::ptx { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_PTX } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE_EXPERIMENTAL _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::device::experimental { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE_EXPERIMENTAL } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DRIVER _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::__driver { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_DRIVER } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;

// Namespaces related to <ranges>
#  define _LIBCUDACXX_BEGIN_NAMESPACE_RANGES _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std::ranges { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_RANGES } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_VIEWS _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std::ranges::views { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_VIEWS } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;

#  define _LIBCUDACXX_BEGIN_NAMESPACE_CPO(_CPO) namespace _CPO {
#  define _LIBCUDACXX_END_NAMESPACE_CPO }

// Namespaces related to chrono / filesystem
#  define _LIBCUDACXX_BEGIN_NAMESPACE_FILESYSTEM _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std { inline namespace __fs { namespace filesystem { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_FILESYSTEM } } } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;

// Shorthands for different qualifiers
  // Namespaces related to execution
#  define _LIBCUDACXX_BEGIN_NAMESPACE_EXECUTION _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std::execution { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_EXECUTION } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;

#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_EXECUTION _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda { namespace execution { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_EXECUTION } } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace hip = cuda;

// Namespace to avoid name collisions with CPOs on clang-16 (see https://godbolt.org/z/9TadonrdM for example)
#if _CCCL_COMPILER(CLANG, ==, 16)
#  define _LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE namespace __hidden {
#  define _LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(_CLASS) } using __hidden::_CLASS;
#else // ^^^ _CCCL_COMPILER(CLANG, ==, 16) ^^^ / vvv !_CCCL_COMPILER(CLANG, ==, 16) vvv
#  define _LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE
#  define _LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(_CLASS)
#endif // !_CCCL_COMPILER(CLANG, ==, 16)

  // Shorthands for different qualifiers
#  define _CUDA_VSTD_NOVERSION ::cuda::std
#  define _CUDA_VSTD           ::cuda::std::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_DEVICE         ::cuda::device::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VRANGES        ::cuda::std::ranges::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VIEWS          ::cuda::std::ranges::views::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VMR            ::cuda::mr::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VPTX           ::cuda::ptx::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VSTD_FS        ::cuda::std::__fs::filesystem::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_STD_EXEC       ::cuda::std::execution::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_EXEC           ::cuda::execution::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_DRIVER         ::cuda::__driver::_LIBCUDACXX_ABI_NAMESPACE

// clang-format on

#endif // _LIBCUDACXX___INTERNAL_NAMESPACES_H
