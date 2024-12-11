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

//UNSUPPORTED: c++11, nvrtc && nvcc-12.0, nvrtc && nvcc-12.1

#include <hip/std/mdspan>
#include <hip/std/cassert>

template<size_t N, class Extents>
__host__ __device__ void test_mdspan_size(hip::std::array<char,N>& storage, Extents&& e)
{
    using extents_type = hip::std::remove_cv_t<hip::std::remove_reference_t<Extents>>;
    hip::std::mdspan<char, extents_type> m(storage.data(), hip::std::forward<Extents>(e));

    static_assert(hip::std::is_same<decltype(m.size()), size_t>::value,
                  "The return type of mdspan::size() must be size_t.");

    // m.size() must not overflow, as long as the product of extents
    // is representable as a value of type size_t.
    assert( m.size() == N );
}


int main(int, char**)
{
    // TEST(TestMdspan, MdspanSizeReturnTypeAndPrecondition)
    {
        hip::std::array<char,12*11> storage;

        static_assert(hip::std::numeric_limits<int8_t>::max() == 127, "max int8_t != 127");
        test_mdspan_size(storage, hip::std::extents<int8_t, 12, 11>{}); // 12 * 11 == 132
    }

    {
        hip::std::array<char,16*17> storage;

        static_assert(hip::std::numeric_limits<uint8_t>::max() == 255, "max uint8_t != 255");
        test_mdspan_size(storage, hip::std::extents<uint8_t, 16, 17>{}); // 16 * 17 == 272
    }

    return 0;
}
