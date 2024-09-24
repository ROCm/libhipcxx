//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hip/std/tuple>

template <class> struct TestLayoutCtors;
template <class Mapping, size_t... DynamicSizes>
struct TestLayoutCtors<hip::std::tuple<
    Mapping,
    hip::std::integer_sequence<size_t, DynamicSizes...>
>>
{
    using mapping_type = Mapping;
    using extents_type = typename mapping_type::extents_type;
    Mapping map = { extents_type{ DynamicSizes... } };
};

template<class T> __host__ __device__ void typed_test_default_ctor()
// TYPED_TEST( TestLayoutCtors, default_ctor )
{
    // Default constructor ensures extents() == Extents() is true.
    using TestFixture = TestLayoutCtors<T>;
    auto m  = typename TestFixture::mapping_type();
    assert( m .extents() == typename TestFixture::extents_type() );
    auto m2 = typename TestFixture::mapping_type{};
    assert( m2.extents() == typename TestFixture::extents_type{} );
    assert( m == m2 );
}

template <class> struct TestLayoutCompatCtors;
template <class Mapping, size_t... DynamicSizes, class Mapping2, size_t... DynamicSizes2>
struct TestLayoutCompatCtors<hip::std::tuple<
  Mapping,
  hip::std::integer_sequence<size_t, DynamicSizes...>,
  Mapping2,
  hip::std::integer_sequence<size_t, DynamicSizes2...>
>> {
  using mapping_type1 = Mapping;
  using mapping_type2 = Mapping2;
  using extents_type1 = hip::std::remove_reference_t<decltype(hip::std::declval<mapping_type1>().extents())>;
  using extents_type2 = hip::std::remove_reference_t<decltype(hip::std::declval<mapping_type2>().extents())>;
  Mapping  map1 = { extents_type1{ DynamicSizes...  } };
  Mapping2 map2 = { extents_type2{ DynamicSizes2... } };
};

template<class T> __host__ __device__ void typed_test_compatible()
//TYPED_TEST(TestLayout{Left|Right}CompatCtors, compatible_construct_{1|2}) {
//TYPED_TEST(TestLayout{Left|Right}CompatCtors, compatible_assign_{1|2}) {
{
    using TestFixture = TestLayoutCompatCtors<T>;

    // Construct
    {
        TestFixture t;

        auto m1 = typename TestFixture::mapping_type1(t.map2);
        assert( m1.extents() == t.map2.extents() );

        auto m2 = typename TestFixture::mapping_type2(t.map1);
        assert( m2.extents() == t.map1.extents() );
    }

    // Assign
    {
        TestFixture t;

#if __MDSPAN_HAS_CXX_17
        if constexpr ( hip::std::is_convertible<typename TestFixture::mapping_type2, typename TestFixture::mapping_type1>::value )
        {
            t.map1 = t.map2;
        }
        else
        {
            t.map1 = typename TestFixture::mapping_type1( t.map2 );
        }
#else
        t.map1 = typename TestFixture::mapping_type1( t.map2 );
#endif

        assert( t.map1.extents() == t.map2.extents() );
    }
}

template<class T> __host__ __device__ void typed_test_compare()
{
    using TestFixture = TestLayoutCompatCtors<T>;

    {
        TestFixture t;

        auto m1 = typename TestFixture::mapping_type1(t.map2);
        assert( m1 == t.map2 );

        auto m2 = typename TestFixture::mapping_type2(t.map1);
        assert( m2 == t.map1 );
    }
}

template <size_t... Ds>
using _sizes = hip::std::integer_sequence<size_t, Ds...>;
template <size_t... Ds>
using _exts  = hip::std::extents<size_t,Ds...>;

template <class E1, class S1, class E2, class S2>
using test_left_type_pair = hip::std::tuple<
  typename hip::std::layout_left::template mapping<E1>, S1,
  typename hip::std::layout_left::template mapping<E2>, S2
>;

template <class E1, class S1, class E2, class S2>
using test_right_type_pair = hip::std::tuple<
    typename hip::std::layout_right::template mapping<E1>, S1,
    typename hip::std::layout_right::template mapping<E2>, S2
>;

template< class T1, class T2, class = void >
struct is_cons_avail : hip::std::false_type {};

template< class T1, class T2 >
struct is_cons_avail< T1
                    , T2
                    , hip::std::enable_if_t< hip::std::is_same< decltype( T1{ hip::std::declval<T2>() } )
                                                                , T1
                                                                >::value
                                            >
                    > : hip::std::true_type {};

template< class T1, class T2 >
constexpr bool is_cons_avail_v = is_cons_avail< T1, T2 >::value;

template< class, class T, class... Indicies >
struct is_paren_op_avail : hip::std::false_type {};

template< class T, class... Indicies >
struct is_paren_op_avail< hip::std::enable_if_t< hip::std::is_same< decltype(hip::std::declval<T>()(hip::std::declval<Indicies>()...))
                                                                    , typename T::index_type
                                                                    >::value
                                                >
                        , T
                        , Indicies...
                        > : hip::std::true_type {};

template< class T, class... Indicies >
constexpr bool is_paren_op_avail_v = is_paren_op_avail< void, T, Indicies... >::value;

template< class T, class RankType, class = void >
struct is_stride_avail : hip::std::false_type {};

template< class T, class RankType >
struct is_stride_avail< T
                      , RankType
                      , hip::std::enable_if_t< hip::std::is_same< decltype( hip::std::declval<T>().stride( hip::std::declval<RankType>() ) )
                                                                  , typename T::index_type
                                                                  >::value
                                              >
                      > : hip::std::true_type {};

template< class T, class RankType >
constexpr bool is_stride_avail_v = is_stride_avail< T, RankType >::value;

// Workaround for variables that are only used in static_assert's
template< typename T >
__host__ __device__ constexpr bool unused( T && ) { return true; }
