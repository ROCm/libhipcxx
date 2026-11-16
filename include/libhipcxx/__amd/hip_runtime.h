// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

// NOTE(HIP/AMD): HIP runtime headers and the few HIP-side definitions the
// library needs. Included from <cuda/std/detail/__config>.
#if !defined(__HIPCC_RTC__)
#  include <hip/hip_runtime.h>
#endif // !__HIPCC_RTC__
// NOTE(HIP/AMD): Under HIPRTC, the HIPRTC-specific runtime stub definitions
// (hipPointerAttribute_t, hipMemoryType, hipError_t, etc.) are provided by the
// test infrastructure header (test/utils/amd/hiprtc/hiprtcc_common.h) which
// prepends them to every compiled source.
#include <libhipcxx/__amd/amd_utils.h>

#ifndef HIPRT_CB
#  define HIPRT_CB
#endif

// NOTE(HIP/AMD): the CUDA driver launch-config aggregate has no layout-
// compatible HIP type (hipLaunchConfig_t uses .stream/.dynamicSmemBytes and
// dim3 grid/block), so <cuda/__launch/*> fills a CUDA-shaped
// ::__libhipcxx::__launch_config that ::cuda::__driver::__launchKernel converts
// to the HIP module-launch call. Define the CUDA-shaped aggregates here (subset
// actually used).
// NOTE(HIP/AMD): this is a host-only launch path; under HIPRTC (device-only RTC)
// hipStream_t is not provided by the COMGR sandbox, so skip the block there.
#if !defined(_LIBHIPCXX_HAS_LAUNCH_CONFIG) && !defined(__HIPCC_RTC__)
#  define _LIBHIPCXX_HAS_LAUNCH_CONFIG
namespace __libhipcxx
{
enum __launch_attribute_id
{
  __launch_attribute_ignore            = 0,
  __launch_attribute_cluster_dimension = 1, // Hopper-only; AMDGCN has no clusters
  __launch_attribute_cooperative       = 2,
  __launch_attribute_priority          = 3,
};
struct __launch_attribute_value
{
  struct
  {
    unsigned int x, y, z;
  } clusterDim;
  int cooperative;
  int priority;
};
struct __launch_attribute
{
  __launch_attribute_id id;
  __launch_attribute_value value;
};
struct __launch_config
{
  unsigned int gridDimX, gridDimY, gridDimZ;
  unsigned int blockDimX, blockDimY, blockDimZ;
  unsigned int sharedMemBytes;
  hipStream_t hStream;
  __launch_attribute* attrs;
  unsigned int numAttrs;
};
} // namespace __libhipcxx
#endif // _LIBHIPCXX_HAS_LAUNCH_CONFIG

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
// NOTE(HIP/AMD): pull <hip/hip_fp8.h> in here (same pattern as bf16/fp16
// above) so that '__hip_fp8_e4m3' / '__hip_fp8_e5m2' are complete types by
// the time they reach a use site like 'is_same_v<_RawTp, __hip_fp8_e4m3>' in
// <cuda/std/__type_traits/num_bits.h>. fp4 / fp6 are intentionally NOT
// pulled in here -- their <hip/hip_fp{4,6}.h> headers have an internal-helper
// symbol clash when included together.
#include <hip/hip_fp8.h>
// NOTE(HIP/AMD): double -> __half conversion (HIP has no __double2half).
// Mark 'inline' to avoid ODR violations when multiple TUs that include this
// header are linked together (the
// c2h-on-HIP test executables hit this -- catch2_runner.cpp.o,
// catch2_runner_helper.cu.o and the per-test .cu.o all otherwise
// emit their own copy of this function).
namespace __libhipcxx
{
__host__ __device__ inline __half __double2half(const double& __value) noexcept
{
  return __float2half(static_cast<float>(__value));
}
} // namespace __libhipcxx

