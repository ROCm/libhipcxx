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
//
// This is a dummy feature that prevents this test from running by default.
// REQUIRES: template-cost-testing

// Test the cost of the mechanism used to create an overload set used by variant
// to determine which alternative to construct.

// The table below compares the compile time and object size for each of the
// variants listed in the RUN script.
//
//  Impl           Compile Time  Object Size
// -----------------------------------------------------
// flat:              959 ms        792 KiB
// recursive:      23,444 ms     23,000 KiB
// -----------------------------------------------------
// variant_old:    16,894 ms     17,000 KiB
// variant_new:     1,105 ms        828 KiB


// RUN: %cxx %flags %compile_flags -std=c++17 -c %s \
// RUN:    -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace -g \
// RUN:    -DTEST_NS=flat_impl -o %S/flat.o
// RUN: %cxx %flags %compile_flags -std=c++17 -c %s \
// RUN:    -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace -g \
// RUN:    -DTEST_NS=rec_impl -o %S/rec.o
// RUN: %cxx %flags %compile_flags -std=c++17 -c %s \
// RUN:    -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace -g \
// RUN:    -DTEST_NS=variant_impl -o %S/variant.o

#include <hip/std/type_traits>
#include <hip/std/tuple>
#include <hip/std/cassert>
#include <hip/std/variant>

#include "test_macros.h"
#include "template_cost_testing.h"

template <size_t Idx>
struct TestType {};

template <class T>
struct ID {
  using type = T;
};

namespace flat_impl {

struct OverloadBase { void operator()() const; };

template <class Tp, size_t Idx>
struct Overload {
  auto operator()(Tp, Tp) const -> ID<Tp>;
};

template <class ...Bases>
struct AllOverloads : OverloadBase, Bases... {};

template <class IdxSeq>
struct MakeOverloads;

template <size_t ..._Idx>
struct MakeOverloads<hip::std::__tuple_indices<_Idx...> > {
  template <class ...Types>
  using Apply = AllOverloads<Overload<Types, _Idx>...>;
};

template <class ...Types>
using Overloads = typename MakeOverloads<
    hip::std::__make_indices_imp<sizeof...(Types), 0> >::template Apply<Types...>;

} // namespace flat_impl


namespace rec_impl {

template <class... Types> struct Overload;

template <>
struct Overload<> { void operator()() const; };

template <class Tp, class... Types>
struct Overload<Tp, Types...> : Overload<Types...> {
  using Overload<Types...>::operator();
  auto operator()(Tp, Tp) const -> ID<Tp>;
};

template <class... Types>
using Overloads = Overload<Types...>;

} // namespace rec_impl

namespace variant_impl {
  template <class ...Types>
  using Overloads = hip::std::__variant_detail::_MakeOverloads<Types...>;
} // naamespace variant_impl

#ifndef TEST_NS
#error TEST_NS must be defined
#endif

#define TEST_TYPE() TestType< __COUNTER__ >,
using T1 = TEST_NS::Overloads<REPEAT_1000(TEST_TYPE) TestType<1>, TestType<1>, int>;
static_assert(__COUNTER__ >= 1000, "");

void fn1(T1 x) { DoNotOptimize(&x); }
void fn2(typename hip::std::invoke_result_t<T1, int, int>::type x) { DoNotOptimize(&x); }

int main() {
  DoNotOptimize(&fn1);
  DoNotOptimize(&fn2);
}
