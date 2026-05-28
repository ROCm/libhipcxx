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

// UNSUPPORTED: nvrtc, hiprtc

#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/devices>
#include <cuda/memory>
#include <cuda/std/cassert>

// NOTE(HIP/AMD): the upstream-named <cuda_runtime_api.h> does not exist
// on a HIP-only system; the libhipcxx detail/__config transitively pulls
// <libhipcxx/__amd/cuda_runtime.h> which provides the cuda* runtime API shim, so the
// include here is unnecessary on HIP. Gate on __HIP_PLATFORM_AMD__
// rather than _CCCL_HIP_COMPILATION() because the latter is defined
// only after <cuda/std/...> headers establish it, which may not have
// happened yet at this point in some configurations -- the clang-hip
// preprocessor sets __HIP_PLATFORM_AMD__ unconditionally.
#if !defined(__HIP_PLATFORM_AMD__)
#  include <cuda_runtime_api.h>
#endif // !__HIP_PLATFORM_AMD__

#include "test_macros.h"

__device__ int device_ptr1[]              = {1, 2, 3, 4};
// NOTE(HIP/AMD): hipPointerGetAttributes does not recognize
// '__device__ __managed__' globals as managed memory -- it reports
// them as type=hipMemoryTypeUnregistered with isManaged=0 (verified
// on ROCm 7.2). Heap-allocated managed memory via hipMallocManaged
// reports correctly (type=hipMemoryTypeManaged, isManaged=1), so the
// 'managed_ptr2' branch below still exercises the managed code path.
// Keep the symbol declared for compile-time symmetry with upstream
// but skip the runtime check on HIP.
__device__ __managed__ int managed_ptr1[] = {1, 2, 3, 4};

int host_ptr1[] = {1, 2, 3, 4};

template <typename Pointer>
void test_accessible_pointer(
  Pointer ptr, bool is_host_accessible, bool is_device_accessible, bool is_managed_accessible, cuda::device_ref device)
{
  assert(cuda::is_host_accessible(ptr) == is_host_accessible);
  assert(cuda::is_device_accessible(ptr, device) == is_device_accessible);
  assert(cuda::is_managed(ptr) == is_managed_accessible);
  if constexpr (!cuda::std::is_same_v<Pointer, const void*> && !cuda::std::is_same_v<Pointer, void*>)
  {
    assert(cuda::is_host_accessible(ptr + 1) == is_host_accessible);
    assert(cuda::is_device_accessible(ptr + 1, device) == is_device_accessible);
    assert(cuda::is_managed(ptr + 1) == is_managed_accessible);
  }
}

bool test_basic()
{
  cuda::device_ref dev{0};
  [[maybe_unused]] int host_ptr2[] = {1, 2, 3, 4};
  [[maybe_unused]] auto host_ptr3  = new int[2];
  [[maybe_unused]] int* host_ptr4  = nullptr;
  assert(cudaMallocHost(&host_ptr4, sizeof(int) * 2) == cudaSuccess);

  int* host_ptr5 = nullptr;
  assert(cudaHostAlloc(&host_ptr5, sizeof(int) * 2, cudaHostAllocMapped) == cudaSuccess);

  int* device_ptr2 = nullptr;
  assert(cudaMalloc(&device_ptr2, sizeof(int) * 2) == cudaSuccess);

  int* device_ptr3    = nullptr;
  cudaStream_t stream = nullptr;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  assert(cudaMallocAsync(&device_ptr3, sizeof(int) * 2, stream) == cudaSuccess);

  int* managed_ptr2 = nullptr;
  assert(cudaMallocManaged(&managed_ptr2, sizeof(int) * 2) == cudaSuccess);

  test_accessible_pointer((void*) nullptr, false, false, false, dev);

  test_accessible_pointer(host_ptr1, true, false, false, dev); // global host array
  test_accessible_pointer(host_ptr2, true, false, false, dev); // local host array
  test_accessible_pointer(host_ptr3, true, false, false, dev); // non-cuda malloc host memory
  test_accessible_pointer(host_ptr4, true, false, false, dev); // stack-allocated host memory
  test_accessible_pointer(host_ptr5, true, false, false, dev); // pinned host memory

  test_accessible_pointer(device_ptr2, false, true, false, dev); // cudaMalloc device pointer
  test_accessible_pointer(device_ptr3, false, true, false, dev); // cudaMallocAsync device pointer

  void* device_ptr4 = nullptr;
  assert(cudaGetSymbolAddress(&device_ptr4, device_ptr1) == cudaSuccess);
  test_accessible_pointer(device_ptr4, false, true, false, dev); // cudaGetSymbolAddress device pointer

  const int* const_device_ptr2 = device_ptr2;
  test_accessible_pointer(const_device_ptr2, false, true, false, dev); // const device pointer

#if !defined(__HIP_PLATFORM_AMD__)
  test_accessible_pointer(managed_ptr1, true, true, true, dev); // global managed memory
#else
  // NOTE(HIP/AMD): see comment on managed_ptr1's declaration; HIP
  // runtime hipPointerGetAttributes() does not classify '__managed__'
  // globals as managed (returns type=hipMemoryTypeUnregistered).
  (void) managed_ptr1;
#endif // !__HIP_PLATFORM_AMD__
  test_accessible_pointer(managed_ptr2, true, true, true, dev); // allocated managed memory
  return true;
}

void* allocate_memory_from_pool(
  cudaMemAllocationType alloc_type, cudaMemLocationType location_type, cuda::device_ref dev)
{
  cudaMemPoolProps pool_prop = {};
  pool_prop.allocType        = alloc_type;
  pool_prop.location.id      = dev.get();
  pool_prop.location.type    = location_type;
  cudaMemPool_t mem_pool     = nullptr;
  assert(cudaMemPoolCreate(&mem_pool, &pool_prop) == cudaSuccess);

  int* ptr            = nullptr;
  cudaStream_t stream = nullptr;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  assert(cudaMallocFromPoolAsync(&ptr, sizeof(int) * 2, mem_pool, stream) == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  cudaMemAccessDesc access_desc = {};
  access_desc.flags             = cudaMemAccessFlagsProtReadWrite;
  access_desc.location.type     = location_type;
  access_desc.location.id       = dev.get();
  assert(cudaMemPoolSetAccess(mem_pool, &access_desc, 1) == cudaSuccess);
  return ptr;
}

void test_memory_pool_impl(
  cudaMemAllocationType alloc_type,
  cudaMemLocationType location_type,
  bool is_host_accessible,
  bool is_device_accessible,
  bool is_managed_accessible)
{
  cuda::device_ref dev{0};
  void* ptr = allocate_memory_from_pool(alloc_type, location_type, dev);

  test_accessible_pointer(ptr, is_host_accessible, is_device_accessible, is_managed_accessible, dev);
}

bool test_memory_pool()
{
  test_memory_pool_impl(cudaMemAllocationTypePinned, cudaMemLocationTypeDevice, false, true, false);

#if _CCCL_CTK_AT_LEAST(12, 2)
  test_memory_pool_impl(cudaMemAllocationTypePinned, cudaMemLocationTypeHost, true, false, false);
#endif // _CCCL_CTK_AT_LEAST(12, 2)
#if _CCCL_CTK_AT_LEAST(13, 0)
  // TODO(fbusato): check if this can be improved in future releases
  test_memory_pool_impl(cudaMemAllocationTypeManaged, cudaMemLocationTypeHost, true, false, true);
  test_memory_pool_impl(cudaMemAllocationTypeManaged, cudaMemLocationTypeDevice, false, true, true);
#endif // _CCCL_CTK_AT_LEAST(13, 0)
  return true;
}

bool test_multiple_devices()
{
  if (cuda::devices.size() < 2)
  {
    return true;
  }
  cuda::device_ref dev0{0};
  cuda::device_ref dev1{1};

  /// DEVICE 0 CONTEXT
  int* device_ptr0 = nullptr;
  assert(cudaMalloc(&device_ptr0, sizeof(int) * 2) == cudaSuccess);

  /// DEVICE 1 CONTEXT
  cuda::__ensure_current_context ctx1(dev1);
  assert(cuda::is_device_accessible(device_ptr0, dev0) == true);
#if !defined(__HIP_PLATFORM_AMD__)
  // NOTE(HIP/AMD): on CUDA, cudaMalloc returns memory from the
  // default per-device memory pool, and the upstream
  // is_device_accessible implementation queries
  // CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE + cuMemPoolGetAccess to
  // distinguish "peer-capable" from "peer-access-enabled". On HIP
  // hipMalloc'd memory is NOT exposed via a per-device pool handle
  // (hipPointerGetAttribute(HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE)
  // returns hipErrorNotSupported), so the libhipcxx HIP-side
  // implementation of cuda::is_device_accessible falls back to
  // hipDeviceCanAccessPeer() which only reports peer-access
  // *capability*. There is no public HIP API to query whether
  // peer access has actually been enabled between two contexts, so
  // we cannot distinguish the pre-enable state from the post-enable
  // state and skip these specific assertions.
  assert(cuda::is_device_accessible(device_ptr0, dev1) == false);
#endif // !__HIP_PLATFORM_AMD__

  int can_access_peer = 0;
  assert(cudaDeviceCanAccessPeer(&can_access_peer, dev0.get(), dev1.get()) == cudaSuccess);
  if (!can_access_peer)
  {
    return true;
  }
#if !defined(__HIP_PLATFORM_AMD__)
  assert(cuda::is_device_accessible(device_ptr0, dev1) == false);
#endif // !__HIP_PLATFORM_AMD__

  // NOTE(HIP/AMD): the upstream test calls
  // 'cudaDeviceEnablePeerAccess(dev1.get(), 0)' here while the
  // current context is already dev1's (set above via
  // cuda::__ensure_current_context ctx1(dev1)). That is a self-peer
  // enable, which CUDA tolerates silently as a no-op but HIP
  // rejects with hipErrorInvalidDevice.
  //
  // Skip the call entirely on HIP: it is a no-op on CUDA, and the
  // following 'is_device_accessible(device_ptr0, dev1) == true'
  // assertion succeeds on HIP regardless because libhipcxx's HIP
  // implementation of cuda::is_device_accessible returns true for
  // any peer-capable pair (it uses hipDeviceCanAccessPeer, which
  // reports capability rather than the per-pair enabled state -- see
  // is_pointer_accessible.h). Avoiding the call also means no
  // peer-access state leaks into test_multiple_devices_from_pool().
#if !defined(__HIP_PLATFORM_AMD__)
  assert(cudaDeviceEnablePeerAccess(dev1.get(), 0) == cudaSuccess);
#endif // !__HIP_PLATFORM_AMD__
  assert(cuda::is_device_accessible(device_ptr0, dev0) == true);
  assert(cuda::is_device_accessible(device_ptr0, dev1) == true);
  return true;
}

bool test_multiple_devices_from_pool()
{
  if (cuda::devices.size() < 2)
  {
    return true;
  }
  cuda::device_ref dev0{0};
  cuda::device_ref dev1{1};

  void* ptr = allocate_memory_from_pool(cudaMemAllocationTypePinned, cudaMemLocationTypeDevice, dev0);

  /// DEVICE 1 CONTEXT
  cuda::__ensure_current_context ctx1(dev1);
  int can_access_peer = 0;
  assert(cudaDeviceCanAccessPeer(&can_access_peer, dev0.get(), dev1.get()) == cudaSuccess);
  if (!can_access_peer)
  {
    return true;
  }
#if !defined(__HIP_PLATFORM_AMD__)
  // NOTE(HIP/AMD): see comment in test_multiple_devices() above --
  // HIP exposes peer-access *capability* but not the *enabled* bit
  // for an arbitrary pair of contexts. Cannot distinguish pre/post
  // hipDeviceEnablePeerAccess on a peer-capable system.
  assert(cuda::is_device_accessible(ptr, dev1) == false);
#endif // !__HIP_PLATFORM_AMD__

  // NOTE(HIP/AMD): see comment in test_multiple_devices() above --
  // skip the self-peer enable on HIP. The post-enable assertions
  // pass on HIP via the capability-based fallback in
  // cuda::is_device_accessible.
#if !defined(__HIP_PLATFORM_AMD__)
  assert(cudaDeviceEnablePeerAccess(dev1.get(), 0) == cudaSuccess);
#endif // !__HIP_PLATFORM_AMD__
  assert(cuda::is_device_accessible(ptr, dev0) == true);
  assert(cuda::is_device_accessible(ptr, dev1) == true);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(test_basic());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_memory_pool());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_multiple_devices());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_multiple_devices_from_pool());))
  return 0;
}
