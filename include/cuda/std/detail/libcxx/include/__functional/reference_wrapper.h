// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#ifndef _LIBCUDACXX___FUNCTIONAL_REFERENCE_WRAPPER_H
#define _LIBCUDACXX___FUNCTIONAL_REFERENCE_WRAPPER_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__functional/weak_result_type.h"
#include "../__memory/addressof.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
class _LIBCUDACXX_TEMPLATE_VIS reference_wrapper : public __weak_result_type<_Tp>
{
public:
    // types
    typedef _Tp type;
private:
    type* __f_;

    static _LIBCUDACXX_INLINE_VISIBILITY void __fun(_Tp&) _NOEXCEPT;
    static void __fun(_Tp&&) = delete;

public:
    template <class _Up, class = __enable_if_t<!__is_same_uncvref<_Up, reference_wrapper>::value, decltype(__fun(declval<_Up>())) > >
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    reference_wrapper(_Up&& __u) _NOEXCEPT_(noexcept(__fun(declval<_Up>()))) {
        type& __f = static_cast<_Up&&>(__u);
        __f_ = _CUDA_VSTD::addressof(__f);
    }

    // access
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    operator type&() const _NOEXCEPT {return *__f_;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    type& get() const _NOEXCEPT {return *__f_;}

    // invoke
    template <class... _ArgTypes>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    typename __invoke_of<type&, _ArgTypes...>::type
    operator() (_ArgTypes&&... __args) const {
        return _CUDA_VSTD::__invoke(get(), _CUDA_VSTD::forward<_ArgTypes>(__args)...);
    }
};

#if _LIBCUDACXX_STD_VER > 14 && !defined(_LIBCUDACXX_HAS_NO_DEDUCTION_GUIDES)
template <class _Tp>
_LIBCUDACXX_HOST_DEVICE reference_wrapper(_Tp&) -> reference_wrapper<_Tp>;
#endif

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
reference_wrapper<_Tp>
ref(_Tp& __t) _NOEXCEPT
{
    return reference_wrapper<_Tp>(__t);
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
reference_wrapper<_Tp>
ref(reference_wrapper<_Tp> __t) _NOEXCEPT
{
    return __t;
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
reference_wrapper<const _Tp>
cref(const _Tp& __t) _NOEXCEPT
{
    return reference_wrapper<const _Tp>(__t);
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
reference_wrapper<const _Tp>
cref(reference_wrapper<_Tp> __t) _NOEXCEPT
{
    return __t;
}

template <class _Tp> void ref(const _Tp&&) = delete;
template <class _Tp> void cref(const _Tp&&) = delete;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_REFERENCE_WRAPPER_H
