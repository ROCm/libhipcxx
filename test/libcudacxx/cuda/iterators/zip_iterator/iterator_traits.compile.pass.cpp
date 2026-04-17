//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Iterator traits and member typedefs in zip_view::<iterator>.

#include <cuda/iterator>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

#if !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC)
#  include <iterator>
#endif // !TEST_COMPILER(NVRTC)

template <class T>
_CCCL_CONCEPT HasIterCategory = _CCCL_REQUIRES_EXPR((T))(typename(typename T::iterator_category));

struct Foo
{};

template <template <class...> class Traits>
TEST_FUNC void test()
{
  { // Single iterator should have tuple value type
    using Iter       = cuda::zip_iterator<int*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  { // Two iterator should have pair value type
    using Iter       = cuda::zip_iterator<int*, Foo*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int, Foo>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  { // !=2 views should have tuple value_type
    using Iter       = cuda::zip_iterator<int*, Foo*, int*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int, Foo, int>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  { // If one iterator is not random access then the whole zip_iterator is not random access
    using Iter       = cuda::zip_iterator<int*, Foo*, bidirectional_iterator<int*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int, Foo, int>>);
    static_assert(cuda::std::bidirectional_iterator<Iter>);
    static_assert(cuda::std::__has_bidirectional_traversal<Iter>);
  }

  { // If one iterator is not bidirectional_iterator then the whole zip_iterator is not bidirectional_iterator
    using Iter       = cuda::zip_iterator<forward_iterator<int*>, Foo*, bidirectional_iterator<int*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int, Foo, int>>);
    static_assert(cuda::std::forward_iterator<Iter>);
    static_assert(cuda::std::__has_forward_traversal<Iter>);
  }

  { // Nothing here
    using Iter = cuda::zip_iterator<forward_iterator<int*>, cpp20_input_iterator<Foo*>, bidirectional_iterator<int*>>;
    static_assert(!HasIterCategory<Iter>);
    static_assert(cuda::std::__has_input_traversal<Iter>);
  }

  { // nested iterator has the right value type
    using Iter       = cuda::zip_iterator<int*, cuda::zip_iterator<Foo*, int*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(
      cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int, cuda::std::tuple<Foo, int>>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  { // working with proxy iterator cuda::discard_iterator
    using Iter       = cuda::zip_iterator<int*, cuda::discard_iterator>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type,
                                       cuda::std::tuple<int, cuda::discard_iterator::__discard_proxy>>);
    static_assert(cuda::std::is_same_v<typename IterTraits::reference,
                                       cuda::std::tuple<int&, cuda::discard_iterator::__discard_proxy>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  { // working with proxy iterator cuda::tabulate_output_iterator
    using Iter       = cuda::zip_iterator<int*, cuda::tabulate_output_iterator<cuda::std::plus<>, short>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type,
                                       cuda::std::tuple<int, cuda::__tabulate_proxy<cuda::std::plus<>, short>>>);
    static_assert(cuda::std::is_same_v<typename IterTraits::reference,
                                       cuda::std::tuple<int&, cuda::__tabulate_proxy<cuda::std::plus<>, short>>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  { // working with proxy iterator cuda::transform_output_iterator
    using Iter       = cuda::zip_iterator<int*, cuda::transform_output_iterator<cuda::std::plus<>, short*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(
      cuda::std::is_same_v<typename IterTraits::value_type,
                           cuda::std::tuple<int, cuda::__transform_output_proxy<cuda::std::plus<>, short*>>>);
    static_assert(
      cuda::std::is_same_v<typename IterTraits::reference,
                           cuda::std::tuple<int&, cuda::__transform_output_proxy<cuda::std::plus<>, short*>>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  { // working with proxy iterator cuda::transform_input_output_iterator
    using Iter =
      cuda::zip_iterator<int*, cuda::transform_input_output_iterator<cuda::std::negate<>, cuda::std::plus<>, short*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int, int>>);
    static_assert(
      cuda::std::is_same_v<
        typename IterTraits::reference,
        cuda::std::tuple<int&, cuda::__transform_input_output_proxy<cuda::std::negate<>, cuda::std::plus<>, short*>>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }
}

TEST_FUNC void test()
{
  test<cuda::std::iterator_traits>();
#if !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC)
  test<std::iterator_traits>();
#endif // !TEST_COMPILER(NVRTC)
}

int main(int, char**)
{
  return 0;
}
