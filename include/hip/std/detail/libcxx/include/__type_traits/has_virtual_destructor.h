//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_HAS_VIRTUAL_DESTRUCTOR_H
#define _LIBCUDACXX___TYPE_TRAITS_HAS_VIRTUAL_DESTRUCTOR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_HAS_VIRTUAL_DESTRUCTOR) && !defined(_LIBCUDACXX_USE_HAS_VIRTUAL_DESTRUCTOR_FALLBACK)

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS has_virtual_destructor
    : public integral_constant<bool, _LIBCUDACXX_HAS_VIRTUAL_DESTRUCTOR(_Tp)> {};

#else

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS has_virtual_destructor
    : public false_type {};

#endif // defined(_LIBCUDACXX_HAS_VIRTUAL_DESTRUCTOR) && !defined(_LIBCUDACXX_USE_HAS_VIRTUAL_DESTRUCTOR_FALLBACK)

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool has_virtual_destructor_v
    = has_virtual_destructor<_Tp>::value;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_HAS_VIRTUAL_DESTRUCTOR_H
