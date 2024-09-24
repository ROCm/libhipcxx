//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// UNSUPPORTED: c++98, c++03, c++11, c++14

//	We have two macros for checking whether or not the underlying C library
//	 has C11 features:
//		TEST_HAS_C11_FEATURES    - which is defined in "test_macros.h"
//		_LIBCUDACXX_HAS_C11_FEATURES - which is defined in <__config>
//	They should always be the same

#include <hip/std/detail/__config>
#include "test_macros.h"

#ifdef TEST_HAS_C11_FEATURES
# ifndef _LIBCUDACXX_HAS_C11_FEATURES
#  error "TEST_HAS_C11_FEATURES is defined, but _LIBCUDACXX_HAS_C11_FEATURES is not"
# endif
#endif

#ifdef _LIBCUDACXX_HAS_C11_FEATURES
# ifndef TEST_HAS_C11_FEATURES
#  error "_LIBCUDACXX_HAS_C11_FEATURES is defined, but TEST_HAS_C11_FEATURES is not"
# endif
#endif

int main(int, char**) {
  return 0;
}
