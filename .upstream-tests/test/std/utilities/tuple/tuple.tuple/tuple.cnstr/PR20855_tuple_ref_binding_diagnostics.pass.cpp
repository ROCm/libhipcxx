// -*- C++ -*-
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
// -*- C++ -*-

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

// <cuda/std/tuple>

// See llvm.org/PR20855

#include <hip/std/tuple>
#include <hip/std/cassert>
#include "test_macros.h"

#if TEST_HAS_BUILTIN_IDENTIFIER(__reference_binds_to_temporary)
# define ASSERT_REFERENCE_BINDS_TEMPORARY(...) static_assert(__reference_binds_to_temporary(__VA_ARGS__), "")
# define ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(...) static_assert(!__reference_binds_to_temporary(__VA_ARGS__), "")
#else
# define ASSERT_REFERENCE_BINDS_TEMPORARY(...) static_assert(true, "")
# define ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(...) static_assert(true, "")
#endif

template <class Tp>
struct ConvertsTo {
  using RawTp = typename hip::std::remove_cv< typename hip::std::remove_reference<Tp>::type>::type;

  __host__ __device__ operator Tp() const {
    return static_cast<Tp>(value);
  }

  mutable RawTp value;
};

struct Base {};
struct Derived : Base {};


static_assert(hip::std::is_same<decltype("abc"), decltype(("abc"))>::value, "");
// hip::std::string not supported
/*
ASSERT_REFERENCE_BINDS_TEMPORARY(hip::std::string const&, decltype("abc"));
ASSERT_REFERENCE_BINDS_TEMPORARY(hip::std::string const&, decltype(("abc")));
ASSERT_REFERENCE_BINDS_TEMPORARY(hip::std::string const&, const char*&&);
*/
ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(int&, const ConvertsTo<int&>&);
ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(const int&, ConvertsTo<int&>&);
ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(Base&, Derived&);


static_assert(hip::std::is_constructible<int&, hip::std::reference_wrapper<int>>::value, "");
static_assert(hip::std::is_constructible<int const&, hip::std::reference_wrapper<int>>::value, "");

template <class T> struct CannotDeduce {
  using type = T;
};

template <class ...Args>
__host__ __device__ void F(typename CannotDeduce<hip::std::tuple<Args...>>::type const&) {}

__host__ __device__ void compile_tests() {
  {
    F<int, int const&>(hip::std::make_tuple(42, 42));
  }
  {
    F<int, int const&>(hip::std::make_tuple<const int&, const int&>(42, 42));
    hip::std::tuple<int, int const&> t(hip::std::make_tuple<const int&, const int&>(42, 42));
  }
  // hip::std::string not supported
  /*
  {
    auto fn = &F<int, hip::std::string const&>;
    fn(hip::std::tuple<int, hip::std::string const&>(42, hip::std::string("a")));
    fn(hip::std::make_tuple(42, hip::std::string("a")));
  }
  */
  {
    Derived d;
    hip::std::tuple<Base&, Base const&> t(d, d);
  }
  {
    ConvertsTo<int&> ct;
    hip::std::tuple<int, int&> t(42, ct);
  }
}

__host__ __device__ void allocator_tests() {
    // hip::std::allocator not supported
    //hip::std::allocator<void> alloc;
    int x = 42;
    {
        hip::std::tuple<int&> t(hip::std::ref(x));
        assert(&hip::std::get<0>(t) == &x);
        // hip::std::allocator not supported
        /*
        hip::std::tuple<int&> t1(hip::std::allocator_arg, alloc, hip::std::ref(x));
        assert(&hip::std::get<0>(t1) == &x);
        */
    }
    {
        auto r = hip::std::ref(x);
        auto const& cr = r;
        hip::std::tuple<int&> t(r);
        assert(&hip::std::get<0>(t) == &x);
        hip::std::tuple<int&> t1(cr);
        assert(&hip::std::get<0>(t1) == &x);
        // hip::std::allocator not supported
        /*
        hip::std::tuple<int&> t2(hip::std::allocator_arg, alloc, r);
        assert(&hip::std::get<0>(t2) == &x);
        hip::std::tuple<int&> t3(hip::std::allocator_arg, alloc, cr);
        assert(&hip::std::get<0>(t3) == &x);
        */
    }
    {
        hip::std::tuple<int const&> t(hip::std::ref(x));
        assert(&hip::std::get<0>(t) == &x);
        hip::std::tuple<int const&> t2(hip::std::cref(x));
        assert(&hip::std::get<0>(t2) == &x);
        // hip::std::allocator not supported
        /*
        hip::std::tuple<int const&> t3(hip::std::allocator_arg, alloc, hip::std::ref(x));
        assert(&hip::std::get<0>(t3) == &x);
        hip::std::tuple<int const&> t4(hip::std::allocator_arg, alloc, hip::std::cref(x));
        assert(&hip::std::get<0>(t4) == &x);
        */
    }
    {
        auto r = hip::std::ref(x);
        auto cr = hip::std::cref(x);
        hip::std::tuple<int const&> t(r);
        assert(&hip::std::get<0>(t) == &x);
        hip::std::tuple<int const&> t2(cr);
        assert(&hip::std::get<0>(t2) == &x);
        // hip::std::allocator not supported
        /*
        hip::std::tuple<int const&> t3(hip::std::allocator_arg, alloc, r);
        assert(&hip::std::get<0>(t3) == &x);
        hip::std::tuple<int const&> t4(hip::std::allocator_arg, alloc, cr);
        assert(&hip::std::get<0>(t4) == &x);
        */
    }
}


int main(int, char**) {
  compile_tests();
  allocator_tests();

  return 0;
}
