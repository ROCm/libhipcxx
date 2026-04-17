//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
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

// UNSUPPORTED: nvrtc, hiprtc

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include "test_macros.h"

template <class Mapping>
TEST_FUNC void test(Mapping map, size_t expected_size)
{
  assert(map.extents().extent(0) == 42);
  assert(map.extents().extent(1) == 1337);
  assert(map.extents().extent(2) == 7);
  assert(sizeof(Mapping) == expected_size);
}

template <class Mapping>
__global__ void test_kernel(Mapping map, size_t expected_size)
{
  test(map, expected_size);
}

void test()
{
  { // all dynamic
    using extents =
      cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;
    using mapping = cuda::std::layout_right::mapping<extents>;
    mapping map{extents{42, 1337, 7}};
    test(map, sizeof(mapping));
    test_kernel<<<1, 1>>>(map, sizeof(mapping));
  }

  { // middle static
    using extents = cuda::std::extents<size_t, cuda::std::dynamic_extent, 1337, cuda::std::dynamic_extent>;
    using mapping = cuda::std::layout_right::mapping<extents>;
    mapping map{extents{42, 7}};
    test(map, sizeof(mapping));
    test_kernel<<<1, 1>>>(map, sizeof(mapping));
  }

  { // middle dynamic
    using extents = cuda::std::extents<size_t, 42, cuda::std::dynamic_extent, 7>;
    using mapping = cuda::std::layout_right::mapping<extents>;
    mapping map{extents{1337}};
    test(map, sizeof(mapping));
    test_kernel<<<1, 1>>>(map, sizeof(mapping));
  }

  { // all dynamic
    using extents = cuda::std::extents<size_t, 42, 1337, 7>;
    using mapping = cuda::std::layout_right::mapping<extents>;
    mapping map{extents{}};
    test(map, sizeof(mapping));
    test_kernel<<<1, 1>>>(map, sizeof(mapping));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
