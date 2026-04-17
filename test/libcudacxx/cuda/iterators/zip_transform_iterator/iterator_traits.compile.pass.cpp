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
{
  TEST_FUNC constexpr operator int() const noexcept
  {
    return 42;
  }
};

template <template <class...> class Traits>
TEST_FUNC void test()
{
  {
    using Iter       = cuda::zip_transform_iterator<TimesTwo, int*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, int>);
    static_assert(cuda::std::is_same_v<typename IterTraits::reference, int>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  {
    using Iter       = cuda::zip_transform_iterator<Plus, int*, Foo*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, int>);
    static_assert(cuda::std::is_same_v<typename IterTraits::reference, int>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  { // value type is not a reference
    using Iter       = cuda::zip_transform_iterator<ReturnFirstLvalueReference, int*, Foo*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, int>);
    static_assert(cuda::std::is_same_v<typename IterTraits::reference, int&>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  { // value type is not a reference
    using Iter       = cuda::zip_transform_iterator<ReturnFirstRvalueReference, int*, Foo*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, int>);
    static_assert(cuda::std::is_same_v<typename IterTraits::reference, int&&>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  { // If one iterator is not random access then the whole zip_transform_iterator is not random access
    using Iter       = cuda::zip_transform_iterator<ReturnFirstRvalueReference, bidirectional_iterator<int*>, Foo*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, int>);
    static_assert(cuda::std::is_same_v<typename IterTraits::reference, int&&>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::bidirectional_iterator<Iter>);
  }

  { // If one iterator is not bidirectional_iterator then the whole zip_transform_iterator is not bidirectional_iterator
    using Iter       = cuda::zip_transform_iterator<ReturnFirstRvalueReference, forward_iterator<int*>, Foo*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, int>);
    static_assert(cuda::std::is_same_v<typename IterTraits::reference, int&&>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::forward_iterator<Iter>);
  }

  { // If one iterator is not forward_iterator then the whole zip_transform_iterator is not forward_iterator
    using Iter =
      cuda::zip_transform_iterator<ReturnFirstRvalueReference, forward_iterator<int*>, cpp20_input_iterator<Foo*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, int>);
    static_assert(cuda::std::is_same_v<typename IterTraits::reference, int&&>);
    static_assert(cuda::std::input_iterator<Iter>);
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
