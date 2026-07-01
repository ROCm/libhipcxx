//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/concepts>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter, class Fn>
_CCCL_CONCEPT HasIterCategory =
  _CCCL_REQUIRES_EXPR((Iter, Fn))(typename(typename cuda::transform_iterator<Iter, Fn>::iterator_category));

#if !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC)
#  include <iterator>
#endif // !TEST_COMPILER(NVRTC)

template <template <class...> class Traits>
__host__ __device__ constexpr bool test()
{
  {
    using Iter       = cuda::transform_iterator<Increment, int*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::reference, int&>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }
  {
    // Member typedefs for random access iterator.
    using Iter       = cuda::transform_iterator<Increment, random_access_iterator<int*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::reference, int&>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }
  {
    // Member typedefs for random access iterator, LWG3798 rvalue reference.
    using Iter       = cuda::transform_iterator<IncrementRvalueRef, random_access_iterator<int*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::reference, int&&>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }
  {
    // Member typedefs for random access iterator/not-lvalue-ref.
    using Iter       = cuda::transform_iterator<PlusOneMutable, random_access_iterator<int*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::reference, int>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }
  {
    // Member typedefs for bidirectional iterator.
    using Iter       = cuda::transform_iterator<Increment, bidirectional_iterator<int*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::reference, int&>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::bidirectional_iterator<Iter>);
    static_assert(cuda::std::__has_bidirectional_traversal<Iter>);
  }
  {
    // Member typedefs for forward iterator.
    using Iter       = cuda::transform_iterator<Increment, forward_iterator<int*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::reference, int&>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::forward_iterator<Iter>);
    static_assert(cuda::std::__has_forward_traversal<Iter>);
  }
  { // Nopthing to do here
    using Iter = cuda::transform_iterator<Increment, cpp17_input_iterator<int*>>;
    static_assert(!HasIterCategory<Increment, cpp17_input_iterator<int*>>);
    static_assert(cuda::std::__has_input_traversal<Iter>);
  }

  {
    // Ensure we can work with other cuda iterators
    using Iter       = cuda::transform_iterator<TimesTwo, cuda::counting_iterator<int>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::reference, int>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    // Ensure we can work with other cuda iterators
    using Iter       = cuda::std::reverse_iterator<cuda::transform_iterator<TimesTwo, cuda::counting_iterator<int>>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::reference, int>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  return true;
}

__host__ __device__ void test()
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
