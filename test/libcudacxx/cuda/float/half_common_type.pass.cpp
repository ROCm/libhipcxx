//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

// NOTE(HIP/AMD): the upstream-named <cuda_bf16.h> / <cuda_fp16.h>
// headers do not exist on a HIP-only system; pull in the HIP-named
// equivalents which define __nv_bfloat16 / __half via the libhipcxx
// HIP bridge. Gate on __HIP_PLATFORM_AMD__ rather than
// _CCCL_HIP_COMPILATION() because the latter is defined only after
// <cuda/std/...> headers establish it, which has not happened yet at
// this point -- the clang-hip preprocessor sets __HIP_PLATFORM_AMD__
// unconditionally.
#if defined(__HIP_PLATFORM_AMD__)
#  include <hip/hip_bf16.h>
#  include <hip/hip_fp16.h>
#else
#  include <cuda_bf16.h>
#  include <cuda_fp16.h>
#endif // !__HIP_PLATFORM_AMD__

#include <cuda/std/type_traits>

#include "test_macros.h"

#if _LIBCUDACXX_HAS_NVFP16()
static_assert(cuda::std::is_same<cuda::std::common_type<__half, __half>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half, __half&>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&, __half>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half, __half&&>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&&, __half>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&, __half&&>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&&, __half&>::type, __half>::value, "");

static_assert(cuda::std::is_same<cuda::std::common_type<__half, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half, float&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half, float&&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&&, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&, float&&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&&, float&>::type, float>::value, "");
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, __nv_bfloat16>::type, __nv_bfloat16>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, __nv_bfloat16&>::type, __nv_bfloat16>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&, __nv_bfloat16>::type, __nv_bfloat16>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, __nv_bfloat16&&>::type, __nv_bfloat16>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&&, __nv_bfloat16>::type, __nv_bfloat16>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&, __nv_bfloat16&&>::type, __nv_bfloat16>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&&, __nv_bfloat16&>::type, __nv_bfloat16>::value,
              "");

static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, float&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, float&&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&&, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&, float&&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&&, float&>::type, float>::value, "");

static_assert(!cuda::std::__has_common_type<__nv_bfloat16, __half>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16, __half&>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16&, __half>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16, __half&&>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16&&, __half>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16&, __half&&>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16&&, __half&>, "");

#endif // _LIBCUDACXX_HAS_NVBF16()

int main(int argc, char** argv)
{
  return 0;
}
