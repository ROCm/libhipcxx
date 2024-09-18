//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
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

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: nvrtc
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

// <cuda/std/tuple>

// template <class T, class Tuple> constexpr T make_from_tuple(Tuple&&);

#include <hip/std/tuple>
#include <hip/std/array>
#include <hip/std/utility>
#if defined(_LIBCUDACXX_HAS_STRING)
#include <hip/std/string>
#endif
#include <hip/std/cassert>

#include "test_macros.h"
#include "type_id.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

template <class Tuple>
struct ConstexprConstructibleFromTuple {
  template <class ...Args>
  __host__ __device__ explicit constexpr ConstexprConstructibleFromTuple(Args&&... xargs)
      : args{hip::std::forward<Args>(xargs)...} {}
  Tuple args;
};

template <class TupleLike>
struct ConstructibleFromTuple;

template <template <class ...> class Tuple, class ...Types>
struct ConstructibleFromTuple<Tuple<Types...>> {
  template <class ...Args>
  __host__ __device__ explicit ConstructibleFromTuple(Args&&... xargs)
      : args(xargs...),
        arg_types(&makeArgumentID<Args&&...>())
  {}
  Tuple<hip::std::decay_t<Types>...> args;
  TypeID const* arg_types;
};

template <class Tp, size_t N>
struct ConstructibleFromTuple<hip::std::array<Tp, N>> {
template <class ...Args>
  __host__ __device__ explicit ConstructibleFromTuple(Args&&... xargs)
      : args{xargs...},
        arg_types(&makeArgumentID<Args&&...>())
  {}
  hip::std::array<Tp, N> args;
  TypeID const* arg_types;
};

template <class Tuple>
__host__ __device__ constexpr bool do_constexpr_test(Tuple&& tup) {
    using RawTuple = hip::std::decay_t<Tuple>;
    using Tp = ConstexprConstructibleFromTuple<RawTuple>;
    return hip::std::make_from_tuple<Tp>(hip::std::forward<Tuple>(tup)).args == tup;
}

// Needed by do_forwarding_test() since it compares pairs of different types.
template <class T1, class T2, class U1, class U2>
__host__ __device__ inline bool operator==(const hip::std::pair<T1, T2>& lhs, const hip::std::pair<U1, U2>& rhs) {
    return lhs.first == rhs.first && lhs.second == rhs.second;
}

template <class ...ExpectTypes, class Tuple>
__host__ __device__ bool do_forwarding_test(Tuple&& tup) {
    using RawTuple = hip::std::decay_t<Tuple>;
    using Tp = ConstructibleFromTuple<RawTuple>;
    const Tp value = hip::std::make_from_tuple<Tp>(hip::std::forward<Tuple>(tup));
    return value.args == tup
        && value.arg_types == &makeArgumentID<ExpectTypes...>();
}

__host__ __device__ void test_constexpr_construction() {
    {
        constexpr hip::std::tuple<> tup;
        static_assert(do_constexpr_test(tup), "");
    }
    {
        constexpr hip::std::tuple<int> tup(42);
        static_assert(do_constexpr_test(tup), "");
    }
    {
        constexpr hip::std::tuple<int, long, void*> tup(42, 101, nullptr);
        static_assert(do_constexpr_test(tup), "");
    }
    {
        constexpr hip::std::pair<int, const char*> p(42, "hello world");
        static_assert(do_constexpr_test(p), "");
    }
    {
        using Tuple = hip::std::array<int, 3>;
        using ValueTp = ConstexprConstructibleFromTuple<Tuple>;
        constexpr Tuple arr = {42, 101, -1};
        constexpr ValueTp value = hip::std::make_from_tuple<ValueTp>(arr);
        static_assert(value.args[0] == arr[0] && value.args[1] == arr[1]
            && value.args[2] == arr[2], "");
    }
}

__host__ __device__ void test_perfect_forwarding() {
    {
        using Tup = hip::std::tuple<>;
        Tup tup;
        Tup const& ctup = tup;
        assert(do_forwarding_test<>(tup));
        assert(do_forwarding_test<>(ctup));
    }
    {
        using Tup = hip::std::tuple<int>;
        Tup tup(42);
        Tup const& ctup = tup;
        assert(do_forwarding_test<int&>(tup));
        assert(do_forwarding_test<int const&>(ctup));
        assert(do_forwarding_test<int&&>(hip::std::move(tup)));
        assert(do_forwarding_test<int const&&>(hip::std::move(ctup)));
    }
    {
        using Tup = hip::std::tuple<int&, const char*, unsigned&&>;
        int x = 42;
        unsigned y = 101;
        Tup tup(x, "hello world", hip::std::move(y));
        Tup const& ctup = tup;
        assert((do_forwarding_test<int&, const char*&, unsigned&>(tup)));
        assert((do_forwarding_test<int&, const char* const&, unsigned &>(ctup)));
        assert((do_forwarding_test<int&, const char*&&, unsigned&&>(hip::std::move(tup))));
        assert((do_forwarding_test<int&, const char* const&&, unsigned &&>(hip::std::move(ctup))));
    }
    // test with pair<T, U>
    {
        using Tup = hip::std::pair<int&, const char*>;
        int x = 42;
        Tup tup(x, "hello world");
        Tup const& ctup = tup;
        assert((do_forwarding_test<int&, const char*&>(tup)));
        assert((do_forwarding_test<int&, const char* const&>(ctup)));
        assert((do_forwarding_test<int&, const char*&&>(hip::std::move(tup))));
        assert((do_forwarding_test<int&, const char* const&&>(hip::std::move(ctup))));
    }
    // test with array<T, I>
    {
        using Tup = hip::std::array<int, 3>;
        Tup tup = {42, 101, -1};
        Tup const& ctup = tup;
        assert((do_forwarding_test<int&, int&, int&>(tup)));
        assert((do_forwarding_test<int const&, int const&, int const&>(ctup)));
        assert((do_forwarding_test<int&&, int&&, int&&>(hip::std::move(tup))));
        assert((do_forwarding_test<int const&&, int const&&, int const&&>(hip::std::move(ctup))));
    }
}

__host__ __device__ void test_noexcept() {
    struct NothrowMoveable {
      NothrowMoveable() = default;
      __host__ __device__ NothrowMoveable(NothrowMoveable const&) {}
      __host__ __device__ NothrowMoveable(NothrowMoveable&&) noexcept {}
    };
    struct TestType {
      __host__ __device__ TestType(int, NothrowMoveable) noexcept {}
      __host__ __device__ TestType(int, int, int) noexcept(false) {}
      __host__ __device__ TestType(long, long, long) noexcept {}
    };
    {
        using Tuple = hip::std::tuple<int, NothrowMoveable>;
        Tuple tup; ((void)tup);
        Tuple const& ctup = tup; ((void)ctup);
        ASSERT_NOT_NOEXCEPT(hip::std::make_from_tuple<TestType>(ctup));
        LIBCPP_ASSERT_NOEXCEPT(hip::std::make_from_tuple<TestType>(hip::std::move(tup)));
    }
    {
        using Tuple = hip::std::pair<int, NothrowMoveable>;
        Tuple tup; ((void)tup);
        Tuple const& ctup = tup; ((void)ctup);
        ASSERT_NOT_NOEXCEPT(hip::std::make_from_tuple<TestType>(ctup));
        LIBCPP_ASSERT_NOEXCEPT(hip::std::make_from_tuple<TestType>(hip::std::move(tup)));
    }
    {
        using Tuple = hip::std::tuple<int, int, int>;
        Tuple tup; ((void)tup);
        ASSERT_NOT_NOEXCEPT(hip::std::make_from_tuple<TestType>(tup));
        unused(tup);
    }
    {
        using Tuple = hip::std::tuple<long, long, long>;
        Tuple tup; ((void)tup);
        LIBCPP_ASSERT_NOEXCEPT(hip::std::make_from_tuple<TestType>(tup));
        unused(tup);
    }
    {
        using Tuple = hip::std::array<int, 3>;
        Tuple tup; ((void)tup);
        ASSERT_NOT_NOEXCEPT(hip::std::make_from_tuple<TestType>(tup));
        unused(tup);
    }
    {
        using Tuple = hip::std::array<long, 3>;
        Tuple tup; ((void)tup);
        LIBCPP_ASSERT_NOEXCEPT(hip::std::make_from_tuple<TestType>(tup));
        unused(tup);
    }
}

int main(int, char**)
{
    test_constexpr_construction();
    test_perfect_forwarding();
    test_noexcept();

  return 0;
}
