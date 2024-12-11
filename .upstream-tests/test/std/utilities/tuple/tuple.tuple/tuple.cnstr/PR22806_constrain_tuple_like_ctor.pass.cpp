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



// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class TupleLike>
//   tuple(TupleLike&&);
// template <class Alloc, class TupleLike>
//   tuple(hip::std::allocator_arg_t, Alloc const&, TupleLike&&);

// Check that the tuple-like ctors are properly disabled when the UTypes...
// constructor should be selected. See PR22806.

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"

template <class Tp>
using uncvref_t = typename hip::std::remove_cv<typename hip::std::remove_reference<Tp>::type>::type;

template <class Tuple, class = uncvref_t<Tuple>>
struct IsTuple : hip::std::false_type {};

template <class Tuple, class ...Args>
struct IsTuple<Tuple, hip::std::tuple<Args...>> : hip::std::true_type {};

struct ConstructibleFromTupleAndInt {
  enum State { FromTuple, FromInt, Copied, Moved };
  State state;

  __host__ __device__ ConstructibleFromTupleAndInt(ConstructibleFromTupleAndInt const&) : state(Copied) {}
  __host__ __device__ ConstructibleFromTupleAndInt(ConstructibleFromTupleAndInt &&) : state(Moved) {}

  template <class Tuple, class = typename hip::std::enable_if<IsTuple<Tuple>::value>::type>
  __host__ __device__ explicit ConstructibleFromTupleAndInt(Tuple&&) : state(FromTuple) {}

  __host__ __device__ explicit ConstructibleFromTupleAndInt(int) : state(FromInt) {}
};

struct ConvertibleFromTupleAndInt {
  enum State { FromTuple, FromInt, Copied, Moved };
  State state;

  __host__ __device__ ConvertibleFromTupleAndInt(ConvertibleFromTupleAndInt const&) : state(Copied) {}
  __host__ __device__ ConvertibleFromTupleAndInt(ConvertibleFromTupleAndInt &&) : state(Moved) {}

  template <class Tuple, class = typename hip::std::enable_if<IsTuple<Tuple>::value>::type>
  __host__ __device__ ConvertibleFromTupleAndInt(Tuple&&) : state(FromTuple) {}

  __host__ __device__ ConvertibleFromTupleAndInt(int) : state(FromInt) {}
};

struct ConstructibleFromInt {
  enum State { FromInt, Copied, Moved };
  State state;

  __host__ __device__ ConstructibleFromInt(ConstructibleFromInt const&) : state(Copied) {}
  __host__ __device__ ConstructibleFromInt(ConstructibleFromInt &&) : state(Moved) {}

  __host__ __device__ explicit ConstructibleFromInt(int) : state(FromInt) {}
};

struct ConvertibleFromInt {
  enum State { FromInt, Copied, Moved };
  State state;

  __host__ __device__ ConvertibleFromInt(ConvertibleFromInt const&) : state(Copied) {}
  __host__ __device__ ConvertibleFromInt(ConvertibleFromInt &&) : state(Moved) {}
  __host__ __device__ ConvertibleFromInt(int) : state(FromInt) {}
};

int main(int, char**)
{
    // Test for the creation of dangling references when a tuple is used to
    // store a reference to another tuple as its only element.
    // Ex hip::std::tuple<hip::std::tuple<int>&&>.
    // In this case the constructors 1) 'tuple(UTypes&&...)'
    // and 2) 'tuple(TupleLike&&)' need to be manually disambiguated because
    // when both #1 and #2 participate in partial ordering #2 will always
    // be chosen over #1.
    // See PR22806  and LWG issue #2549 for more information.
    // (https://bugs.llvm.org/show_bug.cgi?id=22806)
    using T = hip::std::tuple<int>;
    // hip::std::allocator not supported
    // hip::std::allocator<int> A;
    { // rvalue reference
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
        T t1(42);
        hip::std::tuple< T&& > t2(hip::std::move(t1));
        assert(&hip::std::get<0>(t2) == &t1);
#endif
    }
    { // const lvalue reference
        T t1(42);

        hip::std::tuple< T const & > t2(t1);
        assert(&hip::std::get<0>(t2) == &t1);

        hip::std::tuple< T const & > t3(static_cast<T const&>(t1));
        assert(&hip::std::get<0>(t3) == &t1);
    }
    { // lvalue reference
        T t1(42);

        hip::std::tuple< T & > t2(t1);
        assert(&hip::std::get<0>(t2) == &t1);
    }
    { // const rvalue reference
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
        T t1(42);

        hip::std::tuple< T const && > t2(hip::std::move(t1));
        assert(&hip::std::get<0>(t2) == &t1);
#endif
    }
    // hip::std::allocator not supported
    /*
    { // rvalue reference via uses-allocator
        T t1(42);
        hip::std::tuple< T&& > t2(hip::std::allocator_arg, A, hip::std::move(t1));
        assert(&hip::std::get<0>(t2) == &t1);
    }
    { // const lvalue reference via uses-allocator
        T t1(42);

        hip::std::tuple< T const & > t2(hip::std::allocator_arg, A, t1);
        assert(&hip::std::get<0>(t2) == &t1);

        hip::std::tuple< T const & > t3(hip::std::allocator_arg, A, static_cast<T const&>(t1));
        assert(&hip::std::get<0>(t3) == &t1);
    }
    { // lvalue reference via uses-allocator
        T t1(42);

        hip::std::tuple< T & > t2(hip::std::allocator_arg, A, t1);
        assert(&hip::std::get<0>(t2) == &t1);
    }
    { // const rvalue reference via uses-allocator
        T const t1(42);
        hip::std::tuple< T const && > t2(hip::std::allocator_arg, A, hip::std::move(t1));
        assert(&hip::std::get<0>(t2) == &t1);
    }
    */
    // Test constructing a 1-tuple of the form tuple<UDT> from another 1-tuple
    // 'tuple<T>' where UDT *can* be constructed from 'tuple<T>'. In this case
    // the 'tuple(UTypes...)' ctor should be chosen and 'UDT' constructed from
    // 'tuple<T>'.
    {
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
        using VT = ConstructibleFromTupleAndInt;
        hip::std::tuple<int> t1(42);
        hip::std::tuple<VT> t2(t1);
        assert(hip::std::get<0>(t2).state == VT::FromTuple);
#endif
    }
    {
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
        using VT = ConvertibleFromTupleAndInt;
        hip::std::tuple<int> t1(42);
        hip::std::tuple<VT> t2 = {t1};
        assert(hip::std::get<0>(t2).state == VT::FromTuple);
#endif
    }
    // Test constructing a 1-tuple of the form tuple<UDT> from another 1-tuple
    // 'tuple<T>' where UDT cannot be constructed from 'tuple<T>' but can
    // be constructed from 'T'. In this case the tuple-like ctor should be
    // chosen and 'UDT' constructed from 'T'
    {
        using VT = ConstructibleFromInt;
        hip::std::tuple<int> t1(42);
        hip::std::tuple<VT> t2(t1);
        assert(hip::std::get<0>(t2).state == VT::FromInt);
    }
    {
        using VT = ConvertibleFromInt;
        hip::std::tuple<int> t1(42);
        hip::std::tuple<VT> t2 = {t1};
        assert(hip::std::get<0>(t2).state == VT::FromInt);
    }

  return 0;
}
