//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// <span>

// template <class ElementType, size_t Extent>
//     span<byte,
//          Extent == dynamic_extent
//              ? dynamic_extent
//              : sizeof(ElementType) * Extent>
//     as_writable_bytes(span<ElementType, Extent> s) noexcept;


#include <hip/std/span>
#include <hip/std/cassert>

#include "test_macros.h"

template<typename Span>
__host__ __device__
void testRuntimeSpan(Span sp)
{
    ASSERT_NOEXCEPT(hip::std::as_writable_bytes(sp));

    auto spBytes = hip::std::as_writable_bytes(sp);
    using SB = decltype(spBytes);
    ASSERT_SAME_TYPE(hip::std::byte, typename SB::element_type);

    if (sp.extent == hip::std::dynamic_extent)
        assert(spBytes.extent == hip::std::dynamic_extent);
    else
        assert(spBytes.extent == sizeof(typename Span::element_type) * sp.extent);

    assert(static_cast<void*>(spBytes.data()) == static_cast<void*>(sp.data()));
    assert(spBytes.size() == sp.size_bytes());
}

struct A{};
__device__ int iArr2[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};

int main(int, char**)
{
    testRuntimeSpan(hip::std::span<int>        ());
    testRuntimeSpan(hip::std::span<long>       ());
    testRuntimeSpan(hip::std::span<double>     ());
    testRuntimeSpan(hip::std::span<A>          ());

    testRuntimeSpan(hip::std::span<int, 0>        ());
    testRuntimeSpan(hip::std::span<long, 0>       ());
    testRuntimeSpan(hip::std::span<double, 0>     ());
    testRuntimeSpan(hip::std::span<A, 0>          ());

    testRuntimeSpan(hip::std::span<int>(iArr2, 1));
    testRuntimeSpan(hip::std::span<int>(iArr2, 2));
    testRuntimeSpan(hip::std::span<int>(iArr2, 3));
    testRuntimeSpan(hip::std::span<int>(iArr2, 4));
    testRuntimeSpan(hip::std::span<int>(iArr2, 5));

    testRuntimeSpan(hip::std::span<int, 1>(iArr2 + 5, 1));
    testRuntimeSpan(hip::std::span<int, 2>(iArr2 + 4, 2));
    testRuntimeSpan(hip::std::span<int, 3>(iArr2 + 3, 3));
    testRuntimeSpan(hip::std::span<int, 4>(iArr2 + 2, 4));
    testRuntimeSpan(hip::std::span<int, 5>(iArr2 + 1, 5));

  return 0;
}
