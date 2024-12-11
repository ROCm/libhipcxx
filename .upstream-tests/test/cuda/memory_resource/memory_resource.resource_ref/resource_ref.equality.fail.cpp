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

// cuda::mr::resource_ref equality

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#include <hip/memory_resource>

#include <hip/std/cassert>
#include <hip/std/cstdint>

template <class T>
struct property_with_value {
  using value_type = T;
};

template <class T>
struct property_without_value {};

template <class... Properties>
struct resource {
  void* allocate(std::size_t, std::size_t) { return &_val; }

  void deallocate(void* ptr, std::size_t, std::size_t) {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  bool operator==(const resource& other) const { return _val == other._val; }
  bool operator!=(const resource& other) const { return _val != other._val; }

  int _val = 0;

  _LIBCUDACXX_TEMPLATE(class Property)
  (requires !cuda::property_with_value<Property> &&
   _CUDA_VSTD::_One_of<Property, Properties...>) //
      friend void get_property(const resource&, Property) noexcept {}

  _LIBCUDACXX_TEMPLATE(class Property)
  (requires cuda::property_with_value<Property>&&
       _CUDA_VSTD::_One_of<Property, Properties...>) //
      friend typename Property::value_type
      get_property(const resource& res, Property) noexcept {
    return res._val;
  }
};

using ref = cuda::mr::resource_ref<property_with_value<int>,
                                   property_with_value<double>,
                                   property_without_value<std::size_t> >;

using different_properties =
    cuda::mr::resource_ref<property_with_value<short>, property_with_value<int>,
                           property_without_value<std::size_t> >;

using res = resource<property_with_value<int>, property_with_value<double>,
                     property_without_value<std::size_t> >;

void test_equality() {
  res input{42};
  res with_equal_value{42};
  res with_different_value{1337};

  // Requires matching properties
  assert(ref{input} == different_properties{with_equal_value});
  assert(ref{input} != different_properties{with_different_value});
}

int main(int, char**) {
    NV_IF_TARGET(NV_IS_HOST,(
        test_equality();
    ))

    return 0;
}
