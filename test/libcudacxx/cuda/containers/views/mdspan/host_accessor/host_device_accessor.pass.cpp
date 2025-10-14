//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/mdspan>

#include "test_macros.h"

__device__ int device_array[]              = {1, 2, 3, 4};
// NOTE(HIP/AMD): __managed__ globals are not detected as managed by
// HIP's hipPointerGetAttributes (see WAR-18 in CHANGELOG_v3.1.md), so
// the previous workaround in host_device_accessor.h that accepted
// hipMemoryTypeUnregistered (and hostPointer==devicePointer) as
// "managed" silently let unregistered/non-managed pointers pass the
// assertion. The strict check now only accepts hipMallocManaged()
// allocations, so __managed__ globals can no longer be used with
// cuda::managed_mdspan on HIP. Skip those paths on HIP.
#if !defined(__HIP_PLATFORM_AMD__)
__device__ __managed__ int managed_array[] = {1, 2, 3, 4};
#endif // !__HIP_PLATFORM_AMD__

using ext_t = cuda::std::extents<int, 4>;

__host__ __device__ void basic_mdspan_access_test()
{
  int array[] = {1, 2, 3, 4};
// <<<<<<< OLD CODE from 5787c7856c (a90f2d6460) - COMMENTED OUT
//   cuda::host_mdspan<int, ext_t> h_md{array, ext_t{}};
//   cuda::device_mdspan<int, ext_t> d_md{device_array, ext_t{}};
// #if !defined(__HIP_PLATFORM_AMD__)
// =======
  [[maybe_unused]] cuda::host_mdspan<int, ext_t> h_md{array, ext_t{}};
  [[maybe_unused]] cuda::device_mdspan<int, ext_t> d_md{device_array, ext_t{}};
// >>>>>>> END NEW CODE (a90f2d6460)
  cuda::managed_mdspan<int, ext_t> m_md{managed_array, ext_t{}};
#endif // !__HIP_PLATFORM_AMD__
  NV_IF_ELSE_TARGET(NV_IS_HOST, (unused(h_md[0]);), unused(d_md[0]);)
#if !defined(__HIP_PLATFORM_AMD__)
  unused(m_md[0]);
#endif // !__HIP_PLATFORM_AMD__
}

__device__ void device_mdspan_access_test()
{
  int array[] = {1, 2, 3, 4};
  cuda::device_mdspan<int, ext_t> d_md{array, ext_t{}};
  unused(d_md[0]);
}

__global__ void test_kernel(cuda::host_mdspan<int, ext_t> md)
{
  cuda::host_mdspan<int, ext_t> h_md2{md};
  unused(h_md2);
}

// NOTE(HIP/AMD): HIPRTC compiles device code only and doesn't support kernel
// launch syntax (<<<>>>) or runtime APIs (cudaDeviceSynchronize).
#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)

void host_mdspan_to_kernel_test()
{
  int array[] = {1, 2, 3, 4};
  cuda::host_mdspan<int, ext_t> h_md{array, ext_t{}};
  test_kernel<<<1, 1>>>(h_md);
  assert(cudaDeviceSynchronize() == cudaSuccess);
}

// Negative test for managed_accessor's pointer detection. Constructs a
// managed_mdspan over various non-managed buffers and verifies that the
// internal validity check rejects them. This guards against
// regressions that broaden the HIP-specific workaround (WAR-18) and
// silently accept unregistered or plain-device pointers as "managed",
// which would let invalid code pass the assertion.
void managed_accessor_negative_test()
{
  using mdspan_t = cuda::managed_mdspan<int, ext_t>;
  mdspan_t::accessor_type acc{};

  // Stack memory: never managed.
  {
    int stack_buf[4] = {1, 2, 3, 4};
    assert(!acc.__detectably_invalid(stack_buf, 0));
  }

  // Heap (malloc): never managed.
  {
    int* heap_buf = static_cast<int*>(::malloc(4 * sizeof(int)));
    assert(heap_buf != nullptr);
    assert(!acc.__detectably_invalid(heap_buf, 0));
    ::free(heap_buf);
  }

#  if defined(__HIP_PLATFORM_AMD__) || defined(_CCCL_HIP_COMPILER)
  // Plain hipMalloc'd device memory: not managed.
  {
    int* dev_buf = nullptr;
    assert(hipMalloc(&dev_buf, 4 * sizeof(int)) == hipSuccess);
    assert(!acc.__detectably_invalid(dev_buf, 0));
    assert(hipFree(dev_buf) == hipSuccess);
  }

  // Positive control: hipMallocManaged() must be detected as managed.
  {
    int* mgd_buf = nullptr;
    assert(hipMallocManaged(&mgd_buf, 4 * sizeof(int)) == hipSuccess);
    assert(acc.__detectably_invalid(mgd_buf, 0));
    assert(hipFree(mgd_buf) == hipSuccess);
  }
#  else
  // Plain cudaMalloc'd device memory: not managed.
  {
    int* dev_buf = nullptr;
    assert(cudaMalloc(&dev_buf, 4 * sizeof(int)) == cudaSuccess);
    assert(!acc.__detectably_invalid(dev_buf, 0));
    assert(cudaFree(dev_buf) == cudaSuccess);
  }

  // Positive control: cudaMallocManaged() must be detected as managed.
  {
    int* mgd_buf = nullptr;
    assert(cudaMallocManaged(&mgd_buf, 4 * sizeof(int)) == cudaSuccess);
    assert(acc.__detectably_invalid(mgd_buf, 0));
    assert(cudaFree(mgd_buf) == cudaSuccess);
  }
#  endif // !__HIP_PLATFORM_AMD__
}

#endif // !_CCCL_COMPILER(NVRTC) && !_CCCL_COMPILER_HIPRTC

int main(int, char**)
{
  basic_mdspan_access_test();
#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
  NV_IF_TARGET(NV_IS_HOST, (host_mdspan_to_kernel_test(); managed_accessor_negative_test();))
#endif // !_CCCL_COMPILER(NVRTC) && !_CCCL_COMPILER_HIPRTC
  return 0;
}
