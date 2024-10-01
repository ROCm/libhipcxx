//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: nvrtc
// UNSUPPORTED: windows

// cuda::get_property

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#include <hip/std/cassert>
#include <hip/memory_resource>

struct prop_with_value {
  using value_type = int;
};
struct prop {};

struct upstream_with_valueless_property {
  friend constexpr void get_property(const upstream_with_valueless_property&, prop) {}
};
static_assert( cuda::std::invocable<decltype(cuda::get_property), upstream_with_valueless_property, prop>, "");
static_assert(!cuda::std::invocable<decltype(cuda::get_property), upstream_with_valueless_property, prop_with_value>, "");

struct upstream_with_stateful_property {
  friend constexpr int get_property(const upstream_with_stateful_property&, prop_with_value) {
    return 42;
  }
};
static_assert(!cuda::std::invocable<decltype(cuda::get_property), upstream_with_stateful_property, prop>, "");
static_assert( cuda::std::invocable<decltype(cuda::get_property), upstream_with_stateful_property, prop_with_value>, "");

struct upstream_with_both_properties {
  friend constexpr void get_property(const upstream_with_both_properties&, prop) {}
  friend constexpr int get_property(const upstream_with_both_properties&, prop_with_value) {
    return 42;
  }
};
static_assert( cuda::std::invocable<decltype(cuda::get_property), upstream_with_both_properties, prop>, "");
static_assert( cuda::std::invocable<decltype(cuda::get_property), upstream_with_both_properties, prop_with_value>, "");

__host__ __device__ constexpr bool test() {
  upstream_with_valueless_property with_valueless{};
  cuda::get_property(with_valueless, prop{});

  upstream_with_stateful_property with_value{};
  assert(cuda::get_property(with_value, prop_with_value{}) == 42);

  upstream_with_both_properties with_both{};
  cuda::get_property(with_both, prop{});
  assert(cuda::get_property(with_both, prop_with_value{}) == 42);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");
  return 0;
}
