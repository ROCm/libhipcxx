// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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

#pragma once

#include <cuda/__cccl_config>

#ifndef TEST_HALF_T
#  if _CCCL_HAS_NVFP16()
#    define TEST_HALF_T() 1
#  else
#    define TEST_HALF_T() 0
#  endif
#endif // TEST_HALF_T

#ifndef TEST_BF_T
#  if _CCCL_HAS_NVBF16()
#    define TEST_BF_T() 1
#  else
#    define TEST_BF_T() 0
#  endif
#endif // TEST_BF_T

#ifndef TEST_INT128
#  if _CCCL_HAS_INT128() && !_CCCL_CUDA_COMPILER(CLANG) // clang-cuda crashes with int128 in generator.cu
#    define TEST_INT128() 1
#  else
#    define TEST_INT128() 0
#  endif
#endif // TEST_INT128

// NOTE(HIP/AMD): on HIP the upstream-named <cuda_fp16.h> /
// <cuda_bf16.h> headers do not exist; pull the HIP-named equivalents
// (which alias __half / __nv_bfloat16 via the libhipcxx HIP bridge).
// The c2h/half.cuh + c2h/bfloat16.cuh helpers themselves are
// thrust-using and unsupported on HIP, so they stay gated out --
// instead provide the bare 'half_t' / 'bfloat16_t' type aliases that
// c2h consumer code (e.g. catch2_test_helper.h) expects.
#if TEST_HALF_T()
#  if _CCCL_HIP_COMPILATION()
#    include <hip/hip_fp16.h>
using half_t = __half;
#  else
#    include <cuda_fp16.h>
#    include <c2h/half.cuh>
#  endif
#endif // TEST_HALF_T()

#if TEST_BF_T()
#  if _CCCL_HIP_COMPILATION()
#    include <hip/hip_bf16.h>
using bfloat16_t = __nv_bfloat16;
#  else
#    include <cuda_bf16.h>
#    include <c2h/bfloat16.cuh>
#  endif
#endif // TEST_BF_T()
