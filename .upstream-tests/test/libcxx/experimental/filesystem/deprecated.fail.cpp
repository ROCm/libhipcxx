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

// REQUIRES: verify-support
// UNSUPPORTED: hipcc
// UNSUPPORTED: c++98, c++03, hipcc

// <experimental/filesystem>

#include <experimental/filesystem>

// NOTE(HIP): For hipcc (and maybe even for nvcc), this deprecation test is problematic since the *cuda*::std::experimental namespace is not declared. clang-verify will therefore see an unexpected error message in the following line.
// As this test is not critical for us, we for now make it unsupported. This should be revisited at a later time before moving to production. 
using namespace cuda::std::experimental::filesystem; // expected-error {{'filesystem' is deprecated: cuda::std::experimental::filesystem has now been deprecated in favor of C++17's cuda::std::filesystem. Please stop using it and start using cuda::std::filesystem. This experimental version will be removed in LLVM 11. You can remove this warning by defining the _LIBCUDACXX_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM macro.}}

int main(int, char**) {
  return 0;
}
