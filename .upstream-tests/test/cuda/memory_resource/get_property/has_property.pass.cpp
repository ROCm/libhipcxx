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

// cuda::has_property, cuda::has_property_with

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#include <hip/memory_resource>

struct prop_with_value {
  using value_type = int;
};
struct prop {};

static_assert(cuda::property_with_value<prop_with_value>, "");
static_assert(!cuda::property_with_value<prop>, "");

struct valid_property {
  friend void get_property(const valid_property&, prop) {}
};
static_assert(!cuda::has_property<valid_property, prop_with_value>, "");
static_assert(cuda::has_property<valid_property, prop>, "");
static_assert(!cuda::has_property_with<valid_property, prop, int>, "");

struct valid_property_with_value {
  friend int get_property(const valid_property_with_value&, prop_with_value) {
    return 42;
  }
};
static_assert(
    cuda::has_property<valid_property_with_value, prop_with_value>, "");
static_assert(!cuda::has_property<valid_property_with_value, prop>, "");
static_assert(cuda::has_property_with<valid_property_with_value,
                                          prop_with_value, int>,
              "");
static_assert(!cuda::has_property_with<valid_property_with_value,
                                           prop_with_value, double>,
              "");

struct derived_from_property : public valid_property {
  friend int get_property(const derived_from_property&, prop_with_value) {
    return 42;
  }
};
static_assert(cuda::has_property<derived_from_property, prop_with_value>,
              "");
static_assert(cuda::has_property<derived_from_property, prop>, "");
static_assert(
    cuda::has_property_with<derived_from_property, prop_with_value, int>,
    "");
static_assert(!cuda::has_property_with<derived_from_property,
                                           prop_with_value, double>,
              "");

int main(int, char**) { return 0; }
