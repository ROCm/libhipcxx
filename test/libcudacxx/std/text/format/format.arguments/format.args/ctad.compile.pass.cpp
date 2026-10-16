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

// <cuda/std/format>

// template<class Context, class... Args>
//   basic_format_args(format-arg-store<Context, Args...>) -> basic_format_args<Context>;

#include <cuda/std/__format_>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

// A global using-directive makes HIP's ::__any ambiguous with cuda::std::__any on Windows.
namespace test_ns
{
using namespace cuda::std;

static_assert(is_same_v<format_args, decltype(basic_format_args{make_format_args(declval<int&>())})>);
#if _CCCL_HAS_WCHAR_T()
static_assert(is_same_v<wformat_args, decltype(basic_format_args{make_wformat_args(declval<int&>())})>);
#endif // _CCCL_HAS_WCHAR_T()
} // namespace test_ns

int main(int, char**)
{
  return 0;
}
