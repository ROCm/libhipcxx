//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#ifndef _LIBCUDACXX___TUPLE_VECTOR_TYPES_H
#define _LIBCUDACXX___TUPLE_VECTOR_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// <<<<<<< OLD CODE from ffb4cb5cea (22b068b84b) - COMMENTED OUT
// // <<<<<<< OLD CODE from 072425cffa (702d3c08ed) - COMMENTED OUT
// // #if _CCCL_HAS_CTK()
// // =======
// #if _CCCL_HAS_CUDA_COMPILER()
// // >>>>>>> END NEW CODE (702d3c08ed)
// =======
#if _CCCL_HAS_CTK()
// >>>>>>> END NEW CODE (22b068b84b)

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wmismatched-tags")

#  include <cuda/std/__fwd/get.h>
#  include <cuda/std/__tuple_dir/structured_bindings.h>
#  include <cuda/std/__tuple_dir/tuple_element.h>
#  include <cuda/std/__tuple_dir/tuple_size.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/__utility/move.h>

// <<<<<<< OLD CODE from 26ad6b2377 (d06d2a1139) - COMMENTED OUT
// // <<<<<<< OLD CODE from 40c5b56c00 (f69d5ba4be) - COMMENTED OUT
// // #  if !_CCCL_CUDA_COMPILATION()
// // #    include <cuda_runtime_api.h>
// // #  endif // !_CCCL_CUDA_COMPILATION()
// //
// // // <<<<<<< OLD CODE from ffb4cb5cea (22b068b84b) - COMMENTED OUT
// // // #  define _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(__name, __type, __size, ...)                      \
// // //     template <>                                                                                    \
// // //     struct tuple_size<__name##__size##__VA_ARGS__> : _CUDA_VSTD::integral_constant<size_t, __size> \
// // //     {};                                                                                            \
// // //                                                                                                    \
// // //     template <size_t _Ip>                                                                          \
// // //     struct tuple_element<_Ip, __name##__size##__VA_ARGS__>                                         \
// // //     {                                                                                              \
// // //       static_assert(_Ip < __size, "tuple_element index out of range");                             \
// // //       using type = __type;                                                                         \
// // // =======
// // =======
// // >>>>>>> END NEW CODE (f69d5ba4be)
// #  define _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(__name, __type, __size)              \
//     template <>                                                                       \
//     struct tuple_size<__name##__size> : _CUDA_VSTD::integral_constant<size_t, __size> \
//     {};                                                                               \
//                                                                                       \
//     template <size_t _Ip>                                                             \
//     struct tuple_element<_Ip, __name##__size>                                         \
//     {                                                                                 \
//       static_assert(_Ip < __size, "tuple_element index out of range");                \
//       using type = __type;                                                            \
// // >>>>>>> END NEW CODE (22b068b84b)
// =======
#  define _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(__name, __type, __size, ...)                      \
    template <>                                                                                    \
    struct tuple_size<__name##__size##__VA_ARGS__> : _CUDA_VSTD::integral_constant<size_t, __size> \
    {};                                                                                            \
                                                                                                   \
    template <size_t _Ip>                                                                          \
    struct tuple_element<_Ip, __name##__size##__VA_ARGS__>                                         \
    {                                                                                              \
      static_assert(_Ip < __size, "tuple_element index out of range");                             \
      using type = __type;                                                                         \
// >>>>>>> END NEW CODE (d06d2a1139)
    };

#  define _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(__name, __type) \
    _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(__name, __type, 1)           \
    _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(__name, __type, 2)           \
    _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(__name, __type, 3)           \
    _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(__name, __type, 4)

#  define _LIBCUDACXX_SPECIALIZE_GET(__name, __base_type)                                                           \
    template <size_t _Ip>                                                                                           \
    _CCCL_API constexpr __base_type& get(__name& __val) noexcept                                                    \
    {                                                                                                               \
      return _CUDA_VSTD::__get_element<_Ip>::template get<__name, __base_type>(__val);                              \
    }                                                                                                               \
    template <size_t _Ip>                                                                                           \
    _CCCL_API constexpr const __base_type& get(const __name& __val) noexcept                                        \
    {                                                                                                               \
      return _CUDA_VSTD::__get_element<_Ip>::template get<__name, __base_type>(__val);                              \
    }                                                                                                               \
    template <size_t _Ip>                                                                                           \
    _CCCL_API constexpr __base_type&& get(__name&& __val) noexcept                                                  \
    {                                                                                                               \
      return _CUDA_VSTD::__get_element<_Ip>::template get<__name, __base_type>(static_cast<__name&&>(__val));       \
    }                                                                                                               \
    template <size_t _Ip>                                                                                           \
    _CCCL_API constexpr const __base_type&& get(const __name&& __val) noexcept                                      \
    {                                                                                                               \
      return _CUDA_VSTD::__get_element<_Ip>::template get<__name, __base_type>(static_cast<const __name&&>(__val)); \
    }

#  define _LIBCUDACXX_SPECIALIZE_GET_VECTOR(__name, __base_type) \
    _LIBCUDACXX_SPECIALIZE_GET(__name##1, __base_type)           \
    _LIBCUDACXX_SPECIALIZE_GET(__name##2, __base_type)           \
    _LIBCUDACXX_SPECIALIZE_GET(__name##3, __base_type)           \
    _LIBCUDACXX_SPECIALIZE_GET(__name##4, __base_type)

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(__HIP_PLATFORM_AMD__)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(char, char)
#else
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(char, signed char)
#endif
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(uchar, unsigned char)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(short, short)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(ushort, unsigned short)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(int, int)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(uint, unsigned int)
_CCCL_SUPPRESS_DEPRECATED_PUSH
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(long, long)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(ulong, unsigned long)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(longlong, long long)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(ulonglong, unsigned long long)
_CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
// <<<<<<< OLD CODE from 71caa2af63 (6bcfd39214) - COMMENTED OUT
// // <<<<<<< OLD CODE from 26ad6b2377 (d06d2a1139) - COMMENTED OUT
// // _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(long, long, 4, _16a)
// // _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(long, long, 4, _32a)
// // _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(ulong, unsigned long, 4, _16a)
// // _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(ulong, unsigned long, 4, _32a)
// // _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(longlong, long long, 4, _16a)
// // _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(longlong, long long, 4, _32a)
// // =======
// _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(ulong, unsigned long, 4, _16a)
// _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(ulong, unsigned long, 4, _32a)
// // >>>>>>> END NEW CODE (d06d2a1139)
// =======
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(long, long, 4, _16a)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(long, long, 4, _32a)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(ulong, unsigned long, 4, _16a)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(ulong, unsigned long, 4, _32a)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(longlong, long long, 4, _16a)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(longlong, long long, 4, _32a)
// >>>>>>> END NEW CODE (6bcfd39214)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(ulonglong, unsigned long long, 4, _16a)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(ulonglong, unsigned long long, 4, _32a)
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(float, float)
_CCCL_SUPPRESS_DEPRECATED_PUSH
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR(double, double)
_CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(double, double, 4, _16a)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(double, double, 4, _32a)
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE(dim, unsigned int, 3)

template <size_t _Ip>
struct __get_element;

template <>
struct __get_element<0>
{
  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr _BaseType& get(_Vec& __val) noexcept
  {
    return __val.x;
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr const _BaseType& get(const _Vec& __val) noexcept
  {
    return __val.x;
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr _BaseType&& get(_Vec&& __val) noexcept
  {
    return static_cast<_BaseType&&>(__val.x);
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr const _BaseType&& get(const _Vec&& __val) noexcept
  {
    return static_cast<const _BaseType&&>(__val.x);
  }
};

template <>
struct __get_element<1>
{
  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr _BaseType& get(_Vec& __val) noexcept
  {
    return __val.y;
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr const _BaseType& get(const _Vec& __val) noexcept
  {
    return __val.y;
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr _BaseType&& get(_Vec&& __val) noexcept
  {
    return static_cast<_BaseType&&>(__val.y);
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr const _BaseType&& get(const _Vec&& __val) noexcept
  {
    return static_cast<const _BaseType&&>(__val.y);
  }
};
template <>
struct __get_element<2>
{
  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr _BaseType& get(_Vec& __val) noexcept
  {
    return __val.z;
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr const _BaseType& get(const _Vec& __val) noexcept
  {
    return __val.z;
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr _BaseType&& get(_Vec&& __val) noexcept
  {
    return static_cast<_BaseType&&>(__val.z);
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr const _BaseType&& get(const _Vec&& __val) noexcept
  {
    return static_cast<const _BaseType&&>(__val.z);
  }
};

template <>
struct __get_element<3>
{
  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr _BaseType& get(_Vec& __val) noexcept
  {
    return __val.w;
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr const _BaseType& get(const _Vec& __val) noexcept
  {
    return __val.w;
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr _BaseType&& get(_Vec&& __val) noexcept
  {
    return static_cast<_BaseType&&>(__val.w);
  }

  template <class _Vec, class _BaseType>
  static _CCCL_API constexpr const _BaseType&& get(const _Vec&& __val) noexcept
  {
    return static_cast<const _BaseType&&>(__val.w);
  }
};
#if defined(__HIP_PLATFORM_AMD__)
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(char, char)
#else
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(char, signed char)
#endif
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(uchar, unsigned char)
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(short, short)
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(ushort, unsigned short)
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(int, int)
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(uint, unsigned int)
_CCCL_SUPPRESS_DEPRECATED_PUSH
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(long, long)
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(ulong, unsigned long)
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(longlong, long long)
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(ulonglong, unsigned long long)
_CCCL_SUPPRESS_DEPRECATED_POP
// <<<<<<< OLD CODE from 71caa2af63 (6bcfd39214) - COMMENTED OUT
// // <<<<<<< OLD CODE from 26ad6b2377 (d06d2a1139) - COMMENTED OUT
// // #  if _CCCL_CTK_AT_LEAST(13, 0)
// // _LIBCUDACXX_SPECIALIZE_GET(long4_16a, long)
// // _LIBCUDACXX_SPECIALIZE_GET(long4_32a, long)
// // _LIBCUDACXX_SPECIALIZE_GET(ulong4_16a, unsigned long)
// // _LIBCUDACXX_SPECIALIZE_GET(ulong4_32a, unsigned long)
// // _LIBCUDACXX_SPECIALIZE_GET(longlong4_16a, long long)
// // _LIBCUDACXX_SPECIALIZE_GET(longlong4_32a, long long)
// // _LIBCUDACXX_SPECIALIZE_GET(ulonglong4_16a, unsigned long long)
// // _LIBCUDACXX_SPECIALIZE_GET(ulonglong4_32a, unsigned long long)
// // #  endif // _CCCL_CTK_AT_LEAST(13, 0)
// // =======
// // >>>>>>> END NEW CODE (d06d2a1139)
// =======
#  if _CCCL_CTK_AT_LEAST(13, 0)
_LIBCUDACXX_SPECIALIZE_GET(long4_16a, long)
_LIBCUDACXX_SPECIALIZE_GET(long4_32a, long)
_LIBCUDACXX_SPECIALIZE_GET(ulong4_16a, unsigned long)
_LIBCUDACXX_SPECIALIZE_GET(ulong4_32a, unsigned long)
_LIBCUDACXX_SPECIALIZE_GET(longlong4_16a, long long)
_LIBCUDACXX_SPECIALIZE_GET(longlong4_32a, long long)
_LIBCUDACXX_SPECIALIZE_GET(ulonglong4_16a, unsigned long long)
_LIBCUDACXX_SPECIALIZE_GET(ulonglong4_32a, unsigned long long)
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
// >>>>>>> END NEW CODE (6bcfd39214)
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(float, float)
_CCCL_SUPPRESS_DEPRECATED_PUSH
_LIBCUDACXX_SPECIALIZE_GET_VECTOR(double, double)
#  if _CCCL_CTK_AT_LEAST(13, 0)
_LIBCUDACXX_SPECIALIZE_GET(double4_16a, double)
_LIBCUDACXX_SPECIALIZE_GET(double4_32a, double)
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
_LIBCUDACXX_SPECIALIZE_GET(dim3, unsigned int)

_LIBCUDACXX_END_NAMESPACE_STD

#  include <cuda/std/__cccl/epilogue.h>

#  undef _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE
#  undef _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR
#  undef _LIBCUDACXX_SPECIALIZE_GET
#  undef _LIBCUDACXX_SPECIALIZE_GET_VECTOR

_CCCL_DIAG_POP

#endif // _CCCL_HAS_CTK()

#endif // _LIBCUDACXX___TUPLE_VECTOR_TYPES_H
