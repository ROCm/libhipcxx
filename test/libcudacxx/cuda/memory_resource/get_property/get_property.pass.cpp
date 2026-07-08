//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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

// UNSUPPORTED: enable-tile
// error: function-to-pointer decay is unsupported in tile code
// error: taking address of a function is unsupported in tile code

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc, hiprtc

// cuda::get_property

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct prop_with_value
{
  using value_type = int;
};
struct prop
{};

struct upstream_with_valueless_property
{
  TEST_FUNC friend constexpr void get_property(const upstream_with_valueless_property&, prop) {}
};

struct upstream_with_stateful_property
{
  TEST_FUNC friend constexpr int get_property(const upstream_with_stateful_property&, prop_with_value)
  {
    return 42;
  }
};

struct upstream_with_both_properties
{
  TEST_FUNC friend constexpr void get_property(const upstream_with_both_properties&, prop) {}
  TEST_FUNC friend constexpr int get_property(const upstream_with_both_properties&, prop_with_value)
  {
    return 42;
  }
};

TEST_FUNC constexpr bool test()
{
  upstream_with_valueless_property with_valueless{};
  get_property(with_valueless, prop{});

  upstream_with_stateful_property with_value{};
  assert(get_property(with_value, prop_with_value{}) == 42);

  upstream_with_both_properties with_both{};
  get_property(with_both, prop{});
  assert(get_property(with_both, prop_with_value{}) == 42);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
