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

// cuda::mr::resource_ref properties

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

namespace properties_test {
static_assert(cuda::property_with_value<property_with_value<int> >, "");
static_assert(
    cuda::property_with_value<property_with_value<struct someStruct> >, "");

static_assert(!cuda::property_with_value<property_without_value<int> >, "");
static_assert(
    !cuda::property_with_value<property_without_value<struct otherStruct> >,
    "");
} // namespace properties_test

namespace resource_test {
template <class... Properties>
struct resource {
  void* allocate(std::size_t, std::size_t) { return nullptr; }

  void deallocate(void* ptr, std::size_t, std::size_t) {}

  bool operator==(const resource& other) const { return true; }
  bool operator!=(const resource& other) const { return false; }

  int _val = 0;

  _LIBCUDACXX_TEMPLATE(class Property)
    (requires (!cuda::property_with_value<Property>) && _CUDA_VSTD::_One_of<Property, Properties...>) //
  friend void get_property(const resource&, Property) noexcept {}

  _LIBCUDACXX_TEMPLATE(class Property)
    (requires cuda::property_with_value<Property> && _CUDA_VSTD::_One_of<Property, Properties...>) //
  friend typename Property::value_type get_property(const resource& res, Property) noexcept {
    return res._val;
  }
};

// Ensure we have the right size
static_assert(sizeof(cuda::mr::resource_ref<property_with_value<short>,
                                            property_with_value<int> >) ==
              (4 * sizeof(void*)), "");
static_assert(sizeof(cuda::mr::resource_ref<property_with_value<short>,
                                            property_without_value<int> >) ==
              (3 * sizeof(void*)), "");
static_assert(sizeof(cuda::mr::resource_ref<property_without_value<short>,
                                            property_with_value<int> >) ==
              (3 * sizeof(void*)), "");
static_assert(sizeof(cuda::mr::resource_ref<property_without_value<short>,
                                            property_without_value<int> >) ==
              (2 * sizeof(void*)), "");

_LIBCUDACXX_TEMPLATE(class Property, class Ref)
  (requires (!cuda::property_with_value<Property>)) //
    int InvokeIfWithValue(const Ref& ref) {
  return -1;
}

_LIBCUDACXX_TEMPLATE(class Property, class Ref)
  (requires cuda::property_with_value<Property>) //
    typename Property::value_type InvokeIfWithValue(const Ref& ref) {
  return get_property(ref, Property{});
}

_LIBCUDACXX_TEMPLATE(class Property, class Ref)
  (requires cuda::property_with_value<Property>) //
    int InvokeIfWithoutValue(const Ref& ref) {
  return -1;
}

_LIBCUDACXX_TEMPLATE(class Property, class Ref)
  (requires (!cuda::property_with_value<Property>)) //
    int InvokeIfWithoutValue(const Ref& ref) {
  get_property(ref, Property{});
  return 1;
}

template <class... Properties>
void test_resource_ref() {
  constexpr int expected_initially = 42;
  resource<Properties...> input{expected_initially};
  cuda::mr::resource_ref<Properties...> ref{input};

  // Check all the potentially stateful properties
  const int properties_with_value[] = {InvokeIfWithValue<Properties>(ref)...};
  const int expected_with_value[] = {
      ((cuda::property_with_value<Properties>) ? expected_initially
                                                   : -1)...};
  for (std::size_t i = 0; i < sizeof...(Properties); ++i) {
    assert(properties_with_value[i] == expected_with_value[i]);
  }

  const int properties_without_value[] = {
      InvokeIfWithoutValue<Properties>(ref)...};
  const int expected_without_value[] = {
      ((cuda::property_with_value<Properties>) ? -1 : 1)...};
  for (std::size_t i = 0; i < sizeof...(Properties); ++i) {
    assert(properties_without_value[i] == expected_without_value[i]);
  }

  constexpr int expected_after_change = 1337;
  input._val = expected_after_change;

  // Check whether we truly get the right value
  const int properties_with_value2[] = {InvokeIfWithValue<Properties>(ref)...};
  const int expected_with_value2[] = {
      ((cuda::property_with_value<Properties>) ? expected_after_change
                                                   : -1)...};
  for (std::size_t i = 0; i < sizeof...(Properties); ++i) {
    assert(properties_with_value2[i] == expected_with_value2[i]);
  }
}

void test_property_forwarding() {
  using res = resource<property_with_value<short>, property_with_value<int> >;
  using ref = cuda::mr::resource_ref<property_with_value<short> >;

  static_assert(cuda::mr::resource_with<res, property_with_value<short>,
                                        property_with_value<int> >, "");
  static_assert(!cuda::mr::resource_with<ref, property_with_value<short>,
                                         property_with_value<int> >, "");

  static_assert(cuda::mr::resource_with<res, property_with_value<short> >, "");
}

void test_resource_ref() {
  // Test some basic combinations of properties w/o state
  test_resource_ref<property_with_value<short>, property_with_value<int> >();
  test_resource_ref<property_with_value<short>, property_without_value<int> >();
  test_resource_ref<property_without_value<short>,
                    property_without_value<int> >();

  // Test duplicated properties
  test_resource_ref<property_with_value<short>, property_with_value<int>,
                    property_with_value<short> >();

  test_resource_ref<property_without_value<short>, property_without_value<int>,
                    property_without_value<short> >();

  // Ensure we only forward requested properties
  test_property_forwarding();
}

} // namespace resource_test

int main(int, char**) {
    NV_IF_TARGET(NV_IS_HOST,(
        resource_test::test_resource_ref();
    ))

    return 0;
}
