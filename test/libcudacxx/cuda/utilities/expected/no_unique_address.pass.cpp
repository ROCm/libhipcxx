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
#include <cuda/std/expected>

#include "test_macros.h"

template <class Value, class Error>
__host__ __device__ void
test(cuda::std::expected<Value, Error> with_value, cuda::std::expected<Value, Error> with_error, size_t expected_size)
{
  assert(with_value.value() == 42);
  assert(with_error.error() == 1337);
  assert(sizeof(cuda::std::expected<Value, Error>) == expected_size);
}

template <class Value, class Error>
__global__ void test_kernel(
  cuda::std::expected<Value, Error> with_value, cuda::std::expected<Value, Error> with_error, size_t expected_size)
{
  test(with_value, with_error, expected_size);
}

template <int Expected>
struct empty
{
  constexpr empty() = default;
  __host__ __device__ constexpr empty(const int val) noexcept
  {
    assert(val == Expected);
  }

  __host__ __device__ friend constexpr bool operator==(const empty&, int val)
  {
    return val == Expected;
  }
};

void test()
{
  { // non-empty payload, non-empty error
    using expect = cuda::std::expected<int, int>;
    expect with_value{cuda::std::in_place, 42};
    expect with_error{cuda::std::unexpect, 1337};
    test(with_value, with_error, sizeof(expect));
    test_kernel<<<1, 1>>>(with_value, with_error, sizeof(expect));
  }

  { // empty payload, non-empty error
    using expect = cuda::std::expected<empty<42>, int>;
    expect with_value{cuda::std::in_place, 42};
    expect with_error{cuda::std::unexpect, 1337};
    test(with_value, with_error, sizeof(expect));
    test_kernel<<<1, 1>>>(with_value, with_error, sizeof(expect));
  }

  { // non-empty payload, empty error
    using expect = cuda::std::expected<int, empty<1337>>;
    expect with_value{cuda::std::in_place, 42};
    expect with_error{cuda::std::unexpect, 1337};
    test(with_value, with_error, sizeof(expect));
    test_kernel<<<1, 1>>>(with_value, with_error, sizeof(expect));
  }

  { // empty payload, non-empty error
    using expect = cuda::std::expected<empty<42>, empty<1337>>;
    expect with_value{cuda::std::in_place, 42};
    expect with_error{cuda::std::unexpect, 1337};
    test(with_value, with_error, sizeof(expect));
    test_kernel<<<1, 1>>>(with_value, with_error, sizeof(cuda::std::expected<empty<42>, empty<1337>>));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
