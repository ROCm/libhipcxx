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

// Mandates: invoke result must fail to compile when used with device lambdas.
// NOTE(HIP): Tests specific behavior when flag __CUDACC_EXTENDED_LAMBDA__ is set.
// Making this test unsupported because there is no equivalent on HIP side.
// On CUDA, the tests expectedly fails with the error:
// cuda/std/detail/libcxx/include/__functional/invoke.h:480:16:
// error: static assertion failed: Attempt to use an extended __device__ lambda in a context 
// that requires querying its return type in host code. Use a named function object, a __host__ __device__ 
// lambda, or cuda::proclaim_return_type instead.
// UNSUPPORTED: nvrtc, hipcc

// <cuda/std/functional>

// result_of<Fn(ArgTypes...)>

#include <hip/std/type_traits>
#include <hip/std/cassert>
#include "test_macros.h"


template <class Ret, class Fn>
__host__ __device__
void test_lambda(Fn &&)
{
    ASSERT_SAME_TYPE(Ret, typename hip::std::result_of<Fn()>::type);

#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(Ret, typename hip::std::invoke_result<Fn>::type);
#endif
}

int main(int, char**)
{
#if defined(__NVCC__)
    { // extended device lambda
    test_lambda<int>([] __device__ () -> int { return 42; });
    test_lambda<double>([] __device__ () -> double { return 42.0; });
    }
#endif

  return 0;
}
