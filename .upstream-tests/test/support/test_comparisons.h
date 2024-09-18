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
//  A set of routines for testing the comparison operators of a type
//
//      XXXX6 tests all six comparison operators
//      XXXX2 tests only op== and op!=
//
//      AssertComparisonsXAreNoexcept       static_asserts that the operations are all noexcept.
//      AssertComparisonsXReturnBool        static_asserts that the operations return bool.
//      AssertComparisonsXConvertibleToBool static_asserts that the operations return something convertible to bool.


#ifndef TEST_COMPARISONS_H
#define TEST_COMPARISONS_H

#include <hip/std/type_traits>
#include "test_macros.h"

//  Test all six comparison operations for sanity
template <class T, class U = T>
__host__ __device__
TEST_CONSTEXPR_CXX14 bool testComparisons6(const T& t1, const U& t2, bool isEqual, bool isLess)
{
    if (isEqual)
        {
        if (!(t1 == t2)) return false;
        if (!(t2 == t1)) return false;
        if ( (t1 != t2)) return false;
        if ( (t2 != t1)) return false;
        if ( (t1  < t2)) return false;
        if ( (t2  < t1)) return false;
        if (!(t1 <= t2)) return false;
        if (!(t2 <= t1)) return false;
        if ( (t1  > t2)) return false;
        if ( (t2  > t1)) return false;
        if (!(t1 >= t2)) return false;
        if (!(t2 >= t1)) return false;
        }
    else if (isLess)
        {
        if ( (t1 == t2)) return false;
        if ( (t2 == t1)) return false;
        if (!(t1 != t2)) return false;
        if (!(t2 != t1)) return false;
        if (!(t1  < t2)) return false;
        if ( (t2  < t1)) return false;
        if (!(t1 <= t2)) return false;
        if ( (t2 <= t1)) return false;
        if ( (t1  > t2)) return false;
        if (!(t2  > t1)) return false;
        if ( (t1 >= t2)) return false;
        if (!(t2 >= t1)) return false;
        }
    else /* greater */
        {
        if ( (t1 == t2)) return false;
        if ( (t2 == t1)) return false;
        if (!(t1 != t2)) return false;
        if (!(t2 != t1)) return false;
        if ( (t1  < t2)) return false;
        if (!(t2  < t1)) return false;
        if ( (t1 <= t2)) return false;
        if (!(t2 <= t1)) return false;
        if (!(t1  > t2)) return false;
        if ( (t2  > t1)) return false;
        if (!(t1 >= t2)) return false;
        if ( (t2 >= t1)) return false;
        }

    return true;
}

//  Easy call when you can init from something already comparable.
template <class T, class Param>
__host__ __device__
TEST_CONSTEXPR_CXX14 bool testComparisons6Values(Param val1, Param val2)
{
    const bool isEqual = val1 == val2;
    const bool isLess  = val1  < val2;

    return testComparisons6(T(val1), T(val2), isEqual, isLess);
}

template <class T, class U = T>
__host__ __device__
void AssertComparisons6AreNoexcept()
{
    ASSERT_NOEXCEPT(hip::std::declval<const T&>() == hip::std::declval<const U&>());
    ASSERT_NOEXCEPT(hip::std::declval<const T&>() != hip::std::declval<const U&>());
    ASSERT_NOEXCEPT(hip::std::declval<const T&>() <  hip::std::declval<const U&>());
    ASSERT_NOEXCEPT(hip::std::declval<const T&>() <= hip::std::declval<const U&>());
    ASSERT_NOEXCEPT(hip::std::declval<const T&>() >  hip::std::declval<const U&>());
    ASSERT_NOEXCEPT(hip::std::declval<const T&>() >= hip::std::declval<const U&>());
}

template <class T, class U = T>
__host__ __device__
void AssertComparisons6ReturnBool()
{
    ASSERT_SAME_TYPE(decltype(hip::std::declval<const T&>() == hip::std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(hip::std::declval<const T&>() != hip::std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(hip::std::declval<const T&>() <  hip::std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(hip::std::declval<const T&>() <= hip::std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(hip::std::declval<const T&>() >  hip::std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(hip::std::declval<const T&>() >= hip::std::declval<const U&>()), bool);
}


template <class T, class U = T>
__host__ __device__
void AssertComparisons6ConvertibleToBool()
{
    static_assert((hip::std::is_convertible<decltype(hip::std::declval<const T&>() == hip::std::declval<const U&>()), bool>::value), "");
    static_assert((hip::std::is_convertible<decltype(hip::std::declval<const T&>() != hip::std::declval<const U&>()), bool>::value), "");
    static_assert((hip::std::is_convertible<decltype(hip::std::declval<const T&>() <  hip::std::declval<const U&>()), bool>::value), "");
    static_assert((hip::std::is_convertible<decltype(hip::std::declval<const T&>() <= hip::std::declval<const U&>()), bool>::value), "");
    static_assert((hip::std::is_convertible<decltype(hip::std::declval<const T&>() >  hip::std::declval<const U&>()), bool>::value), "");
    static_assert((hip::std::is_convertible<decltype(hip::std::declval<const T&>() >= hip::std::declval<const U&>()), bool>::value), "");
}

//  Test all two comparison operations for sanity
template <class T, class U = T>
__host__ __device__
TEST_CONSTEXPR_CXX14 bool testComparisons2(const T& t1, const U& t2, bool isEqual)
{
    if (isEqual)
        {
        if (!(t1 == t2)) return false;
        if (!(t2 == t1)) return false;
        if ( (t1 != t2)) return false;
        if ( (t2 != t1)) return false;
        }
    else /* not equal */
        {
        if ( (t1 == t2)) return false;
        if ( (t2 == t1)) return false;
        if (!(t1 != t2)) return false;
        if (!(t2 != t1)) return false;
        }

    return true;
}

//  Easy call when you can init from something already comparable.
template <class T, class Param>
__host__ __device__
TEST_CONSTEXPR_CXX14 bool testComparisons2Values(Param val1, Param val2)
{
    const bool isEqual = val1 == val2;

    return testComparisons2(T(val1), T(val2), isEqual);
}

template <class T, class U = T>
__host__ __device__
void AssertComparisons2AreNoexcept()
{
    ASSERT_NOEXCEPT(hip::std::declval<const T&>() == hip::std::declval<const U&>());
    ASSERT_NOEXCEPT(hip::std::declval<const T&>() != hip::std::declval<const U&>());
}

template <class T, class U = T>
__host__ __device__
void AssertComparisons2ReturnBool()
{
    ASSERT_SAME_TYPE(decltype(hip::std::declval<const T&>() == hip::std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(hip::std::declval<const T&>() != hip::std::declval<const U&>()), bool);
}


template <class T, class U = T>
__host__ __device__
void AssertComparisons2ConvertibleToBool()
{
    static_assert((hip::std::is_convertible<decltype(hip::std::declval<const T&>() == hip::std::declval<const U&>()), bool>::value), "");
    static_assert((hip::std::is_convertible<decltype(hip::std::declval<const T&>() != hip::std::declval<const U&>()), bool>::value), "");
}

#endif // TEST_COMPARISONS_H
