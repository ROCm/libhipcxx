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

struct resource {
  void* allocate(std::size_t, std::size_t) { return &_val; }

  void deallocate(void* ptr, std::size_t, std::size_t) {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  bool operator==(const resource& other) const { return _val == other._val; }
  bool operator!=(const resource& other) const { return _val != other._val; }

  int _val = 0;
};

void test_allocate() {
  { // allocate(size)
    resource input{42};
    cuda::mr::resource_ref<> ref{input};

    // Ensure that we properly pass on the allocate function
    assert(input.allocate(0, 0) == ref.allocate(0));

    int expected_after_deallocate = 1337;
    ref.deallocate(static_cast<void*>(&expected_after_deallocate), 0);
    assert(input._val == expected_after_deallocate);
  }

  { // allocate(size, alignment)
    resource input{42};
    cuda::mr::resource_ref<> ref{input};

    // Ensure that we properly pass on the allocate function
    assert(input.allocate(0, 0) == ref.allocate(0, 0));

    int expected_after_deallocate = 1337;
    ref.deallocate(static_cast<void*>(&expected_after_deallocate), 0, 0);
    assert(input._val == expected_after_deallocate);
  }
}

int main(int, char**) {
    NV_IF_TARGET(NV_IS_HOST,(
      test_allocate();
    ))

    return 0;
}
