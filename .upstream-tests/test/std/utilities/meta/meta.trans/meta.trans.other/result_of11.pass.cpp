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
//
// UNSUPPORTED: c++98, c++03
//
// TODO: Triage and fix.
// XFAIL: msvc-19.0
//
// <cuda/std/functional>
//
// result_of<Fn(ArgTypes...)>

#define _LIBCUDACXX_ENABLE_CXX20_REMOVED_TYPE_TRAITS
#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/type_traits>
// #include <cuda/std/memory>
// #include <cuda/std/utility>
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(3013) // a volatile function parameter is deprecated

struct wat
{
    __host__ __device__
    wat& operator*() { return *this; }
    __host__ __device__
    void foo();
};

struct F {};
struct FD : public F {};

#if TEST_STD_VER > 11
template <typename T, typename U>
struct test_invoke_result;

template <typename Fn, typename ...Args, typename Ret>
struct test_invoke_result<Fn(Args...), Ret>
{
    __host__ __device__
    static void call()
    {
        static_assert(cuda::std::is_invocable<Fn, Args...>::value, "");
        static_assert(cuda::std::is_invocable_r<Ret, Fn, Args...>::value, "");
        ASSERT_SAME_TYPE(Ret, typename cuda::std::invoke_result<Fn, Args...>::type);
        ASSERT_SAME_TYPE(Ret,        cuda::std::invoke_result_t<Fn, Args...>);
    }
};
#endif

template <class T, class U>
__host__ __device__
void test_result_of_imp()
{
    ASSERT_SAME_TYPE(U, typename cuda::std::result_of<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(U,        cuda::std::result_of_t<T>);
#endif
#if TEST_STD_VER > 11
    test_invoke_result<T, U>::call();
#endif
}

int main(int, char**)
{
    {
    typedef char F::*PMD;
    test_result_of_imp<PMD(F                &), char                &>();
    test_result_of_imp<PMD(F const          &), char const          &>();
    test_result_of_imp<PMD(F volatile       &), char volatile       &>();
    test_result_of_imp<PMD(F const volatile &), char const volatile &>();

    test_result_of_imp<PMD(F                &&), char                &&>();
    test_result_of_imp<PMD(F const          &&), char const          &&>();
    test_result_of_imp<PMD(F volatile       &&), char volatile       &&>();
    test_result_of_imp<PMD(F const volatile &&), char const volatile &&>();

    test_result_of_imp<PMD(F                ), char &&>();
    test_result_of_imp<PMD(F const          ), char &&>();
    test_result_of_imp<PMD(F volatile       ), char &&>();
    test_result_of_imp<PMD(F const volatile ), char &&>();

    test_result_of_imp<PMD(FD                &), char                &>();
    test_result_of_imp<PMD(FD const          &), char const          &>();
    test_result_of_imp<PMD(FD volatile       &), char volatile       &>();
    test_result_of_imp<PMD(FD const volatile &), char const volatile &>();

    test_result_of_imp<PMD(FD                &&), char                &&>();
    test_result_of_imp<PMD(FD const          &&), char const          &&>();
    test_result_of_imp<PMD(FD volatile       &&), char volatile       &&>();
    test_result_of_imp<PMD(FD const volatile &&), char const volatile &&>();

    test_result_of_imp<PMD(FD                ), char &&>();
    test_result_of_imp<PMD(FD const          ), char &&>();
    test_result_of_imp<PMD(FD volatile       ), char &&>();
    test_result_of_imp<PMD(FD const volatile ), char &&>();

#if !(defined(__NVCC__) || defined(__CUDACC_RTC__) || defined(__HIPCC__) || defined(__HIPCC_RTC__))
    test_result_of_imp<PMD(cuda::std::unique_ptr<F>),        char &>();
    test_result_of_imp<PMD(cuda::std::unique_ptr<F const>),  const char &>();
    test_result_of_imp<PMD(cuda::std::unique_ptr<FD>),       char &>();
    test_result_of_imp<PMD(cuda::std::unique_ptr<FD const>), const char &>();

    test_result_of_imp<PMD(cuda::std::reference_wrapper<F>),        char &>();
    test_result_of_imp<PMD(cuda::std::reference_wrapper<F const>),  const char &>();
    test_result_of_imp<PMD(cuda::std::reference_wrapper<FD>),       char &>();
    test_result_of_imp<PMD(cuda::std::reference_wrapper<FD const>), const char &>();
#endif
    }
    {
    test_result_of_imp<int (F::* (F       &)) ()                &, int> ();
    test_result_of_imp<int (F::* (F       &)) () const          &, int> ();
    test_result_of_imp<int (F::* (F       &)) () volatile       &, int> ();
    test_result_of_imp<int (F::* (F       &)) () const volatile &, int> ();
    test_result_of_imp<int (F::* (F const &)) () const          &, int> ();
    test_result_of_imp<int (F::* (F const &)) () const volatile &, int> ();
    test_result_of_imp<int (F::* (F volatile &)) () volatile       &, int> ();
    test_result_of_imp<int (F::* (F volatile &)) () const volatile &, int> ();
    test_result_of_imp<int (F::* (F const volatile &)) () const volatile &, int> ();

    test_result_of_imp<int (F::* (F       &&)) ()                &&, int> ();
    test_result_of_imp<int (F::* (F       &&)) () const          &&, int> ();
    test_result_of_imp<int (F::* (F       &&)) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (F       &&)) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F const &&)) () const          &&, int> ();
    test_result_of_imp<int (F::* (F const &&)) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F volatile &&)) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (F volatile &&)) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F const volatile &&)) () const volatile &&, int> ();

    test_result_of_imp<int (F::* (F       )) ()                &&, int> ();
    test_result_of_imp<int (F::* (F       )) () const          &&, int> ();
    test_result_of_imp<int (F::* (F       )) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (F       )) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F const )) () const          &&, int> ();
    test_result_of_imp<int (F::* (F const )) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F volatile )) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (F volatile )) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F const volatile )) () const volatile &&, int> ();
    }
    {
    test_result_of_imp<int (F::* (FD       &)) ()                &, int> ();
    test_result_of_imp<int (F::* (FD       &)) () const          &, int> ();
    test_result_of_imp<int (F::* (FD       &)) () volatile       &, int> ();
    test_result_of_imp<int (F::* (FD       &)) () const volatile &, int> ();
    test_result_of_imp<int (F::* (FD const &)) () const          &, int> ();
    test_result_of_imp<int (F::* (FD const &)) () const volatile &, int> ();
    test_result_of_imp<int (F::* (FD volatile &)) () volatile       &, int> ();
    test_result_of_imp<int (F::* (FD volatile &)) () const volatile &, int> ();
    test_result_of_imp<int (F::* (FD const volatile &)) () const volatile &, int> ();

    test_result_of_imp<int (F::* (FD       &&)) ()                &&, int> ();
    test_result_of_imp<int (F::* (FD       &&)) () const          &&, int> ();
    test_result_of_imp<int (F::* (FD       &&)) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (FD       &&)) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (FD const &&)) () const          &&, int> ();
    test_result_of_imp<int (F::* (FD const &&)) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (FD volatile &&)) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (FD volatile &&)) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (FD const volatile &&)) () const volatile &&, int> ();

    test_result_of_imp<int (F::* (FD       )) ()                &&, int> ();
    test_result_of_imp<int (F::* (FD       )) () const          &&, int> ();
    test_result_of_imp<int (F::* (FD       )) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (FD       )) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (FD const )) () const          &&, int> ();
    test_result_of_imp<int (F::* (FD const )) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (FD volatile )) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (FD volatile )) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (FD const volatile )) () const volatile &&, int> ();
    }
    {
#if !(defined(__NVCC__) || defined(__CUDACC_RTC__) || defined(__HIPCC__) || defined(__HIPCC_RTC__))
    test_result_of_imp<int (F::* (cuda::std::reference_wrapper<F>))       (),       int>();
    test_result_of_imp<int (F::* (cuda::std::reference_wrapper<const F>)) () const, int>();
    test_result_of_imp<int (F::* (cuda::std::unique_ptr<F>       ))       (),       int>();
    test_result_of_imp<int (F::* (cuda::std::unique_ptr<const F> ))       () const, int>();
#endif
    }
    test_result_of_imp<decltype(&wat::foo)(wat), void>();

  return 0;
}
