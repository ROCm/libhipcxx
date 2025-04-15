//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef __CCCL_ATTRIBUTES_H
#define __CCCL_ATTRIBUTES_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/dialect.h>

#ifdef __has_attribute
#  define _CCCL_HAS_ATTRIBUTE(__x) __has_attribute(__x)
#else // ^^^ __has_attribute ^^^ / vvv !__has_attribute vvv
#  define _CCCL_HAS_ATTRIBUTE(__x) 0
#endif // !__has_attribute

#ifdef __has_cpp_attribute
#  define _CCCL_HAS_CPP_ATTRIBUTE(__x) __has_cpp_attribute(__x)
#else // ^^^ __has_cpp_attribute ^^^ / vvv !__has_cpp_attribute vvv
#  define _CCCL_HAS_CPP_ATTRIBUTE(__x) 0
#endif // !__has_cpp_attribute

#ifdef __has_declspec_attribute
#  define _CCCL_HAS_DECLSPEC_ATTRIBUTE(__x) __has_declspec_attribute(__x)
#else // ^^^ __has_declspec_attribute ^^^ / vvv !__has_declspec_attribute vvv
#  define _CCCL_HAS_DECLSPEC_ATTRIBUTE(__x) 0
#endif // !__has_declspec_attribute

// MSVC needs extra help with empty base classes
#if _CCCL_COMPILER(MSVC) || _CCCL_HAS_DECLSPEC_ATTRIBUTE(empty_bases)
#  define _CCCL_DECLSPEC_EMPTY_BASES __declspec(empty_bases)
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  define _CCCL_DECLSPEC_EMPTY_BASES
#endif // !_CCCL_COMPILER(MSVC)

// <<<<<<< OLD CODE from ff1ebe1588 (b60ee01d87) - COMMENTED OUT
// // Use a function like macro to imply that it must be followed by a semicolon
// #if _CCCL_HAS_CPP_ATTRIBUTE(fallthrough)
// #  define _CCCL_FALLTHROUGH() [[fallthrough]]
// #elif _CCCL_COMPILER(NVRTC) || defined(_CCCL_COMPILER_HIPRTC)
// #  define _CCCL_FALLTHROUGH() ((void) 0)
// #elif _CCCL_HAS_CPP_ATTRIBUTE(clang::fallthrough)
// #  define _CCCL_FALLTHROUGH() [[clang::fallthrough]]
// #elif _CCCL_COMPILER(NVHPC)
// #  define _CCCL_FALLTHROUGH()
// #elif _CCCL_HAS_ATTRIBUTE(fallthrough) || _CCCL_COMPILER(GCC, >=, 7)
// #  define _CCCL_FALLTHROUGH() __attribute__((__fallthrough__))
// #else
// #  define _CCCL_FALLTHROUGH() ((void) 0)
// #endif
//
// =======
// >>>>>>> END NEW CODE (b60ee01d87)
#if _CCCL_HAS_ATTRIBUTE(__nodebug__)
#  define _CCCL_NODEBUG __attribute__((__nodebug__))
#else // ^^^ _CCCL_HAS_ATTRIBUTE(__nodebug__) ^^^ / vvv !_CCCL_HAS_ATTRIBUTE(__nodebug__) vvv
#  define _CCCL_NODEBUG
#endif // !_CCCL_HAS_ATTRIBUTE(__nodebug__)

// The nodebug attribute flattens aliases down to the actual type rather typename meow<T>::type
#if _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_NODEBUG_ALIAS _CCCL_NODEBUG
#else // ^^^ _CCCL_CUDA_COMPILER(CLANG) ^^^ / vvv !_CCCL_CUDA_COMPILER(CLANG) vvv
#  define _CCCL_NODEBUG_ALIAS
#endif // !_CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_COMPILER(MSVC) || _CCCL_HAS_CPP_ATTRIBUTE(no_unique_address) < 201803L
// MSVC implementation has lead to multiple issues with silent runtime corruption when passing data into kernels
#  define _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
#  define _CCCL_NO_UNIQUE_ADDRESS
#elif _CCCL_HAS_CPP_ATTRIBUTE(no_unique_address)
#  define _CCCL_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#  define _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
#  define _CCCL_NO_UNIQUE_ADDRESS
#endif

// Passing objects with nested [[no_unique_address]] to kernels leads to data corruption
// This happens up to clang18
#if !defined(_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS) && _CCCL_COMPILER(CLANG)
#  define _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
#endif // !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS && _CCCL_COMPILER(CLANG)

// It always fails with clang
#if _CCCL_COMPILER(CLANG)
#  define _CCCL_NODISCARD_FRIEND friend
#else
#  define _CCCL_NODISCARD_FRIEND [[nodiscard]] friend
#endif

#if _CCCL_COMPILER(MSVC) // vvv _CCCL_COMPILER(MSVC) vvv
#  define _CCCL_RESTRICT __restrict
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  define _CCCL_RESTRICT __restrict__
#endif // ^^^ !_CCCL_COMPILER(MSVC) ^^^

#if _CCCL_HAS_CPP_ATTRIBUTE(assume)
#  define _CCCL_ASSUME(...) [[assume(__VA_ARGS__)]]
#elif _CCCL_CUDA_COMPILER(NVCC)
#  define _CCCL_ASSUME(...) \
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__builtin_assume(__VA_ARGS__);), (_CCCL_BUILTIN_ASSUME(__VA_ARGS__);))
#else
#  define _CCCL_ASSUME(...) _CCCL_BUILTIN_ASSUME(__VA_ARGS__)
#endif

#if _CCCL_HAS_CPP_ATTRIBUTE(clang::no_specializations)
#  define _CCCL_NO_SPECIALIZATIONS_BECAUSE(_MSG)   [[clang::no_specializations(_MSG)]]
#  define _CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS() 1
#elif _CCCL_HAS_CPP_ATTRIBUTE(msvc::no_specializations)
#  define _CCCL_NO_SPECIALIZATIONS_BECAUSE(_MSG)   [[msvc::no_specializations(_MSG)]]
#  define _CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS() 1
#else // ^^^ has attribute no_specializations ^^^ / vvv hasn't attribute no_specializations vvv
#  define _CCCL_NO_SPECIALIZATIONS_BECAUSE(_MSG)
#  define _CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS() 0
#endif // ^^^ hasn't attribute no_specializations ^^^

#define _CCCL_NO_SPECIALIZATIONS \
  _CCCL_NO_SPECIALIZATIONS_BECAUSE("Users are not allowed to specialize this cccl entity")

#endif // __CCCL_ATTRIBUTES_H
