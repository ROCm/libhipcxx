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

// T shall be an object type and shall satisfy the requirements of Destructible

#include <cuda/std/optional>

using cuda::std::optional;

struct X
{
private:
    ~X() {}
};

int main(int, char**)
{
    using cuda::std::optional;
    {
        // expected-error-re@.*optional:* 2 {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with a reference type is ill-formed}}
        optional<int&> opt1;
        optional<int&&> opt2;
    }
    {
        // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with a non-destructible type is ill-formed}}
        optional<X> opt3;
    }
    {
        // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with a non-object type is undefined behavior}}
        // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with a non-destructible type is ill-formed}}
        optional<void()> opt4;
    }
    {
        // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with a non-object type is undefined behavior}}
        // expected-error-re@.*optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with a non-destructible type is ill-formed}}
        // expected-error@.*optional:* 1+ {{cannot form a reference to 'void'}}
        optional<const void> opt4;
    }
    // FIXME these are garbage diagnostics that Clang should not produce
    // expected-error@.*optional:* 0+ {{is not a base class}}

  return 0;
}
