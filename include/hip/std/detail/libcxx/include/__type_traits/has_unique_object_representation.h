//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_HAS_UNIQUE_OBJECT_REPRESENTATION_H
#define _LIBCUDACXX___TYPE_TRAITS_HAS_UNIQUE_OBJECT_REPRESENTATION_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/remove_all_extents.h"
#include "../__type_traits/remove_cv.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11 && defined(_LIBCUDACXX_HAS_UNIQUE_OBJECT_REPRESENTATIONS)

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS has_unique_object_representations
    : public integral_constant<bool,
       __has_unique_object_representations(remove_cv_t<remove_all_extents_t<_Tp>>)> {};

#if !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool has_unique_object_representations_v = has_unique_object_representations<_Tp>::value;
#endif

#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_HAS_UNIQUE_OBJECT_REPRESENTATION_H
