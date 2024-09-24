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

int main(int, char**)
{
    // C array
    {
        const int d[5] = {1,2,3,4,5};
#if defined (__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
        hip::std::mdspan m(d);
#else
        hip::std::mdspan<const int, hip::std::extents<size_t,5>> m(d);
#endif

        assert( m.data_handle() == d );
    }

    // std array
    {
        hip::std::array<int,5> d = {1,2,3,4,5};
#if defined (__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
        hip::std::mdspan m(d.data());
#else
        hip::std::mdspan<int, hip::std::extents<size_t,5>> m(d.data());
#endif

        assert( m.data_handle() == d.data() );
    }

    // C pointer
    {
        hip::std::array<int,5> d = {1,2,3,4,5};
        int* ptr = d.data();
#if defined (__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
        hip::std::mdspan m(ptr);
#else
        hip::std::mdspan<int, hip::std::extents<size_t,5>> m(ptr);
#endif

        assert( m.data_handle() == ptr );
    }

    // Copy constructor
    {
        hip::std::array<int,5> d = {1,2,3,4,5};
#if defined (__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
        hip::std::mdspan m0(d.data());
        hip::std::mdspan m (m0);
#else
        hip::std::mdspan<int, hip::std::extents<size_t,5>> m0(d.data());
        hip::std::mdspan<int, hip::std::extents<size_t,5>> m (m0);
#endif

        assert( m.data_handle() == m0.data_handle() );
    }

    return 0;
}
