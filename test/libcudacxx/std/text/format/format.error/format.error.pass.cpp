//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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
// NOTE(HIP/AMD): Excludes hiprtc since cuda::std::format_error requires
// <stdexcept>, which under HIPRTC pulls in libstdc++ <cstdint> -> system
// <stdint.h> and conflicts with cuda/std/cstdint type aliases.

// <cuda/std/format>

// class format_error;

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/cstring>
#include <cuda/std/type_traits>

#include <string>

#include "test_macros.h"

void test_format_error()
{
#if __cpp_lib_format >= 201907L
  static_assert(cuda::std::is_same_v<cuda::std::format_error, std::format_error>);
#endif // __cpp_lib_format >= 201907L

  static_assert(cuda::std::is_base_of_v<std::runtime_error, cuda::std::format_error>);
  static_assert(cuda::std::is_polymorphic_v<cuda::std::format_error>);

  {
    const char* msg = "format_error message c-string";
    cuda::std::format_error e(msg);
    assert(cuda::std::strcmp(e.what(), msg) == 0);
    cuda::std::format_error e2(e);
    assert(cuda::std::strcmp(e2.what(), msg) == 0);
    e2 = e;
    assert(cuda::std::strcmp(e2.what(), msg) == 0);
  }
  {
    std::string msg("format_error message std::string");
    cuda::std::format_error e(msg);
    assert(e.what() == msg);
    cuda::std::format_error e2(e);
    assert(e2.what() == msg);
    e2 = e;
    assert(e2.what() == msg);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_format_error();))
  return 0;
}
