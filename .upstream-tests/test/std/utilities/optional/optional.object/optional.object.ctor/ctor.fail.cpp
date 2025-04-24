//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: nvrtc
// <cuda/std/optional>

// T shall be an object type other than cv in_place_t or cv nullopt_t
//   and shall satisfy the Cpp17Destructible requirements.
// Note: array types do not satisfy the Cpp17Destructible requirements.

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

struct NonDestructible { ~NonDestructible() = delete; };

int main(int, char**)
{
    {
    cuda::std::optional<char &> o1;          // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with a reference type is ill-formed}}
    cuda::std::optional<NonDestructible> o2; // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with a non-destructible type is ill-formed}}
    cuda::std::optional<char[20]> o3;        // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with an array type is ill-formed}}
    }

    {
    cuda::std::optional<               cuda::std::in_place_t> o1; // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with in_place_t is ill-formed}}
    cuda::std::optional<const          cuda::std::in_place_t> o2; // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with in_place_t is ill-formed}}
    cuda::std::optional<      volatile cuda::std::in_place_t> o3; // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with in_place_t is ill-formed}}
    cuda::std::optional<const volatile cuda::std::in_place_t> o4; // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with in_place_t is ill-formed}}
    }

    {
    cuda::std::optional<               cuda::std::nullopt_t> o1; // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with nullopt_t is ill-formed}}
    cuda::std::optional<const          cuda::std::nullopt_t> o2; // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with nullopt_t is ill-formed}}
    cuda::std::optional<      volatile cuda::std::nullopt_t> o3; // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with nullopt_t is ill-formed}}
    cuda::std::optional<const volatile cuda::std::nullopt_t> o4; // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with nullopt_t is ill-formed}}
    }

    return 0;
}
