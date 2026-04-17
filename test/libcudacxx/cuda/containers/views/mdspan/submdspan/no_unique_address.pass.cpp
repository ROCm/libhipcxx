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

template <size_t ExpectedSize, class OffsetType, class ExtentType, class StrideType>
TEST_FUNC void test(cuda::std::strided_slice<OffsetType, ExtentType, StrideType> slice, size_t expected_size)
{
  using strided_slice = cuda::std::strided_slice<OffsetType, ExtentType, StrideType>;
  assert(slice.offset == 42);
  assert(slice.extent == 1337);
  assert(slice.stride == 7);
  assert(sizeof(strided_slice) == expected_size);
  static_assert(sizeof(strided_slice) == ExpectedSize, "Size mismatch");
}

template <size_t ExpectedSize, class OffsetType, class ExtentType, class StrideType>
__global__ void test_kernel(cuda::std::strided_slice<OffsetType, ExtentType, StrideType> slice, size_t expected_size)
{
  test<ExpectedSize>(slice, expected_size);
}

void test()
{
  { // all non_empty
    using strided_slice = cuda::std::strided_slice<int, int, int>;
    strided_slice slice{42, 1337, 7};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // OffsetType empty
    using strided_slice = cuda::std::strided_slice<cuda::std::integral_constant<int, 42>, int, int>;
    strided_slice slice{{}, 1337, 7};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // ExtentType empty
    using strided_slice = cuda::std::strided_slice<int, cuda::std::integral_constant<int, 1337>, int>;
    strided_slice slice{42, {}, 7};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // StrideType empty
    using strided_slice = cuda::std::strided_slice<int, int, cuda::std::integral_constant<int, 7>>;
    strided_slice slice{42, 1337, {}};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // OffsetType + StrideType empty
    using strided_slice =
      cuda::std::strided_slice<cuda::std::integral_constant<int, 42>, int, cuda::std::integral_constant<int, 7>>;
    strided_slice slice{{}, 1337, {}};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // All empty
    using strided_slice =
      cuda::std::strided_slice<cuda::std::integral_constant<int, 42>,
                               cuda::std::integral_constant<int, 1337>,
                               cuda::std::integral_constant<int, 7>>;
    strided_slice slice{};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    // cannot call a kernel with an Empty parameter type
    // test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
