//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIP_WAVEFRONT_SIZE_H
#define HIP_WAVEFRONT_SIZE_H

#if defined(__HIP_PLATFORM_AMD__)

/// @brief Get the wavefront size for AMD GPUs at runtime.
///
/// Uses the hardware builtin to query the actual wavefront size,
/// which varies by GPU architecture:
/// - gfx940: 32 threads per wavefront
/// - gfx950: 64 threads per wavefront
/// - gfx1250: 32 threads per wavefront
///
/// @return Wavefront size (32 or 64)
__device__ __host__ inline int get_wavefront_size()
{
#if defined(__HIP_DEVICE_COMPILE__)
  // Device code: query hardware wavefront size
  return __builtin_amdgcn_wavefrontsize();
#else
  // Host code: query device properties for actual wavefront size
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.warpSize;
#endif
}

#else
// Non-AMD platforms: default to 32 (NVIDIA warp size)
__device__ __host__ inline int get_wavefront_size()
{
  return 32;
}
#endif

#endif // HIP_WAVEFRONT_SIZE_H
