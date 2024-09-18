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



// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... Tuples> tuple<CTypes...> tuple_cat(Tuples&&... tpls);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/utility>
// hip::std::string not supported
//#include <hip/std/array>
// hip::std::array not supported
//#include <hip/std/string>
#include <hip/std/cassert>

#include "test_macros.h"
#include "MoveOnly.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

int main(int, char**)
{
    {
        hip::std::tuple<> t = hip::std::tuple_cat();
        unused(t); // Prevent unused warning
    }
    {
        hip::std::tuple<> t1;
        hip::std::tuple<> t2 = hip::std::tuple_cat(t1);
        unused(t2); // Prevent unused warning
    }
    {
        hip::std::tuple<> t = hip::std::tuple_cat(hip::std::tuple<>());
        unused(t); // Prevent unused warning
    }
    // hip::std::array not supported
    /*
    {
        hip::std::tuple<> t = hip::std::tuple_cat(hip::std::array<int, 0>());
        unused(t); // Prevent unused warning
    }
    */
    {
        hip::std::tuple<int> t1(1);
        hip::std::tuple<int> t = hip::std::tuple_cat(t1);
        assert(hip::std::get<0>(t) == 1);
    }

#if TEST_STD_VER > 11
    {
        constexpr hip::std::tuple<> t = hip::std::tuple_cat();
        unused(t); // Prevent unused warning
    }
    {
        constexpr hip::std::tuple<> t1;
        constexpr hip::std::tuple<> t2 = hip::std::tuple_cat(t1);
        unused(t2); // Prevent unused warning
    }
    {
        constexpr hip::std::tuple<> t = hip::std::tuple_cat(hip::std::tuple<>());
        unused(t); // Prevent unused warning
    }
    // hip::std::array not supported
    /*
    {
        constexpr hip::std::tuple<> t = hip::std::tuple_cat(hip::std::array<int, 0>());
        unused(t); // Prevent unused warning
    }
    */
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
    {
        constexpr hip::std::tuple<int> t1(1);
        constexpr hip::std::tuple<int> t = hip::std::tuple_cat(t1);
        static_assert(hip::std::get<0>(t) == 1, "");
    }
    {
        constexpr hip::std::tuple<int> t1(1);
        constexpr hip::std::tuple<int, int> t = hip::std::tuple_cat(t1, t1);
        static_assert(hip::std::get<0>(t) == 1, "");
        static_assert(hip::std::get<1>(t) == 1, "");
    }
#endif
#endif
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
    {
        hip::std::tuple<int, MoveOnly> t =
                                hip::std::tuple_cat(hip::std::tuple<int, MoveOnly>(1, 2));
        assert(hip::std::get<0>(t) == 1);
        assert(hip::std::get<1>(t) == 2);
    }
#endif
    // hip::std::array not supported
    /*
    {
        hip::std::tuple<int, int, int> t = hip::std::tuple_cat(hip::std::array<int, 3>());
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == 0);
        assert(hip::std::get<2>(t) == 0);
    }
    */
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
    {
        hip::std::tuple<int, MoveOnly> t = hip::std::tuple_cat(hip::std::pair<int, MoveOnly>(2, 1));
        assert(hip::std::get<0>(t) == 2);
        assert(hip::std::get<1>(t) == 1);
    }
#endif
    {
        hip::std::tuple<> t1;
        hip::std::tuple<> t2;
        hip::std::tuple<> t3 = hip::std::tuple_cat(t1, t2);
        unused(t3); // Prevent unused warning
    }
    {
        hip::std::tuple<> t1;
        hip::std::tuple<int> t2(2);
        hip::std::tuple<int> t3 = hip::std::tuple_cat(t1, t2);
        assert(hip::std::get<0>(t3) == 2);
    }
    {
        hip::std::tuple<> t1;
        hip::std::tuple<int> t2(2);
        hip::std::tuple<int> t3 = hip::std::tuple_cat(t2, t1);
        assert(hip::std::get<0>(t3) == 2);
    }
    {
        hip::std::tuple<int*> t1;
        hip::std::tuple<int> t2(2);
        hip::std::tuple<int*, int> t3 = hip::std::tuple_cat(t1, t2);
        assert(hip::std::get<0>(t3) == nullptr);
        assert(hip::std::get<1>(t3) == 2);
    }
    {
        hip::std::tuple<int*> t1;
        hip::std::tuple<int> t2(2);
        hip::std::tuple<int, int*> t3 = hip::std::tuple_cat(t2, t1);
        assert(hip::std::get<0>(t3) == 2);
        assert(hip::std::get<1>(t3) == nullptr);
    }
    {
        hip::std::tuple<int*> t1;
        hip::std::tuple<int, double> t2(2, 3.5);
        hip::std::tuple<int*, int, double> t3 = hip::std::tuple_cat(t1, t2);
        assert(hip::std::get<0>(t3) == nullptr);
        assert(hip::std::get<1>(t3) == 2);
        assert(hip::std::get<2>(t3) == 3.5);
    }
    {
        hip::std::tuple<int*> t1;
        hip::std::tuple<int, double> t2(2, 3.5);
        hip::std::tuple<int, double, int*> t3 = hip::std::tuple_cat(t2, t1);
        assert(hip::std::get<0>(t3) == 2);
        assert(hip::std::get<1>(t3) == 3.5);
        assert(hip::std::get<2>(t3) == nullptr);
    }
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
    {
        hip::std::tuple<int*, MoveOnly> t1(nullptr, 1);
        hip::std::tuple<int, double> t2(2, 3.5);
        hip::std::tuple<int*, MoveOnly, int, double> t3 =
                                              hip::std::tuple_cat(hip::std::move(t1), t2);
        assert(hip::std::get<0>(t3) == nullptr);
        assert(hip::std::get<1>(t3) == 1);
        assert(hip::std::get<2>(t3) == 2);
        assert(hip::std::get<3>(t3) == 3.5);
    }
    {
        hip::std::tuple<int*, MoveOnly> t1(nullptr, 1);
        hip::std::tuple<int, double> t2(2, 3.5);
        hip::std::tuple<int, double, int*, MoveOnly> t3 =
                                              hip::std::tuple_cat(t2, hip::std::move(t1));
        assert(hip::std::get<0>(t3) == 2);
        assert(hip::std::get<1>(t3) == 3.5);
        assert(hip::std::get<2>(t3) == nullptr);
        assert(hip::std::get<3>(t3) == 1);
    }
    {
        hip::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        hip::std::tuple<int*, MoveOnly> t2(nullptr, 4);
        hip::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   hip::std::tuple_cat(hip::std::move(t1), hip::std::move(t2));
        assert(hip::std::get<0>(t3) == 1);
        assert(hip::std::get<1>(t3) == 2);
        assert(hip::std::get<2>(t3) == nullptr);
        assert(hip::std::get<3>(t3) == 4);
    }

    {
        hip::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        hip::std::tuple<int*, MoveOnly> t2(nullptr, 4);
        hip::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   hip::std::tuple_cat(hip::std::tuple<>(),
                                                  hip::std::move(t1),
                                                  hip::std::move(t2));
        assert(hip::std::get<0>(t3) == 1);
        assert(hip::std::get<1>(t3) == 2);
        assert(hip::std::get<2>(t3) == nullptr);
        assert(hip::std::get<3>(t3) == 4);
    }
    {
        hip::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        hip::std::tuple<int*, MoveOnly> t2(nullptr, 4);
        hip::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   hip::std::tuple_cat(hip::std::move(t1),
                                                  hip::std::tuple<>(),
                                                  hip::std::move(t2));
        assert(hip::std::get<0>(t3) == 1);
        assert(hip::std::get<1>(t3) == 2);
        assert(hip::std::get<2>(t3) == nullptr);
        assert(hip::std::get<3>(t3) == 4);
    }
    {
        hip::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        hip::std::tuple<int*, MoveOnly> t2(nullptr, 4);
        hip::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   hip::std::tuple_cat(hip::std::move(t1),
                                                  hip::std::move(t2),
                                                  hip::std::tuple<>());
        assert(hip::std::get<0>(t3) == 1);
        assert(hip::std::get<1>(t3) == 2);
        assert(hip::std::get<2>(t3) == nullptr);
        assert(hip::std::get<3>(t3) == 4);
    }
    {
        hip::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        hip::std::tuple<int*, MoveOnly> t2(nullptr, 4);
        hip::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly, int> t3 =
                                   hip::std::tuple_cat(hip::std::move(t1),
                                                  hip::std::move(t2),
                                                  hip::std::tuple<int>(5));
        assert(hip::std::get<0>(t3) == 1);
        assert(hip::std::get<1>(t3) == 2);
        assert(hip::std::get<2>(t3) == nullptr);
        assert(hip::std::get<3>(t3) == 4);
        assert(hip::std::get<4>(t3) == 5);
    }
    {
        // See bug #19616.
        auto t1 = hip::std::tuple_cat(
            hip::std::make_tuple(hip::std::make_tuple(1)),
            hip::std::make_tuple()
        );
        assert(t1 == hip::std::make_tuple(hip::std::make_tuple(1)));

        auto t2 = hip::std::tuple_cat(
            hip::std::make_tuple(hip::std::make_tuple(1)),
            hip::std::make_tuple(hip::std::make_tuple(2))
        );
        assert(t2 == hip::std::make_tuple(hip::std::make_tuple(1), hip::std::make_tuple(2)));
    }
    {
        int x = 101;
        hip::std::tuple<int, const int, int&, const int&, int&&> t(42, 101, x, x, hip::std::move(x));
        const auto& ct = t;
        hip::std::tuple<int, const int, int&, const int&> t2(42, 101, x, x);
        const auto& ct2 = t2;

        auto r = hip::std::tuple_cat(hip::std::move(t), hip::std::move(ct), t2, ct2);

        ASSERT_SAME_TYPE(decltype(r), hip::std::tuple<
            int, const int, int&, const int&, int&&,
            int, const int, int&, const int&, int&&,
            int, const int, int&, const int&,
            int, const int, int&, const int&>);
        unused(r);
    }
#endif
  return 0;
}
