//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

// UNSUPPORTED: hipcc
// NOTE(HIP): This test includes the header cuda/std/mutex which does not exist.
// HIPCC gives the correct error: cuda/std/mutex file not found.
// However, other errors are expected, consider the expected-error. 
// UNSUPPORTED: libcpp-has-no-threads

// [[nodiscard]] on constructors isn't supported by all compilers
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9
// UNSUPPORTED: apple-clang-9, apple-clang-10, apple-clang-11
// UNSUPPORTED: gcc-5

// [[nodiscard]] isn't supported in C++98 and C++03 (not even as an extension)
// UNSUPPORTED: c++98, c++03

// <cuda/std/mutex>

// template <class Mutex> class lock_guard;

// [[nodiscard]] explicit lock_guard(mutex_type& m);
// [[nodiscard]] lock_guard(mutex_type& m, adopt_lock_t);

// Test that we properly apply [[nodiscard]] to lock_guard's constructors,
// which is a libc++ extension.

// MODULES_DEFINES: _LIBCUDACXX_ENABLE_NODISCARD
#define _LIBCUDACXX_ENABLE_NODISCARD
#include <cuda/std/mutex>

int main(int, char**) {
    cuda::std::mutex m;
    cuda::std::lock_guard<cuda::std::mutex>{m}; // expected-error{{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    cuda::std::lock_guard<cuda::std::mutex>{m, cuda::std::adopt_lock}; // expected-error{{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    return 0;
}
