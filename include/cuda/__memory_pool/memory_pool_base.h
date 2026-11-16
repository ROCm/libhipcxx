//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
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

#ifndef _CUDA___MEMORY_RESOURCE_MEMORY_POOL_BASE_H
#define _CUDA___MEMORY_RESOURCE_MEMORY_POOL_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#  include <cuda/__device/attributes.h>
#  include <cuda/__device/device_ref.h>
#  include <cuda/__memory_resource/any_resource.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/internal_streams.h>
#  include <cuda/__stream/stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/cstddef>

#  if _CCCL_HOSTED()
#    include <vector>
#  endif // _CCCL_HOSTED()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-braces")
// clang complains about missing braces in CUmemLocation constructor but GCC complains if we add them

enum class __pool_attr_settable : bool
{
};

template <::hipMemPoolAttr _Attr, typename _Type, __pool_attr_settable _Settable>
struct __pool_attr_impl
{
  using type = _Type;

  [[nodiscard]] _CCCL_HOST_API constexpr operator ::hipMemPoolAttr() const noexcept
  {
    return _Attr;
  }

  [[nodiscard]] _CCCL_HOST_API type operator()(::hipMemPool_t __pool) const
  {
    size_t __value = ::cuda::__driver::__mempoolGetAttribute(__pool, static_cast<::hipMemPoolAttr>(_Attr));
    return static_cast<type>(__value);
  }

  static void set(::hipMemPool_t __pool, type __value)
  {
    size_t __value_copy = __value;
    if constexpr (_Settable == __pool_attr_settable{true})
    {
      ::cuda::__driver::__mempoolSetAttribute(__pool, static_cast<::hipMemPoolAttr>(_Attr), &__value_copy);
    }
    else
    {
      _CCCL_THROW(::std::invalid_argument, "This attribute can't be set");
    }
  }
};

template <::hipMemPoolAttr _Attr>
struct __pool_attr : __pool_attr_impl<_Attr, size_t, __pool_attr_settable{true}>
{};

template <>
struct __pool_attr<::hipMemPoolReuseFollowEventDependencies>
    : __pool_attr_impl<::hipMemPoolReuseFollowEventDependencies, bool, __pool_attr_settable{true}>
{};

template <>
struct __pool_attr<::hipMemPoolReuseAllowOpportunistic>
    : __pool_attr_impl<::hipMemPoolReuseAllowOpportunistic, bool, __pool_attr_settable{true}>
{};

template <>
struct __pool_attr<::hipMemPoolReuseAllowInternalDependencies>
    : __pool_attr_impl<::hipMemPoolReuseAllowInternalDependencies, bool, __pool_attr_settable{true}>
{};

template <>
struct __pool_attr<::hipMemPoolAttrReservedMemCurrent>
    : __pool_attr_impl<::hipMemPoolAttrReservedMemCurrent, size_t, __pool_attr_settable{false}>
{};

template <>
struct __pool_attr<::hipMemPoolAttrUsedMemCurrent>
    : __pool_attr_impl<::hipMemPoolAttrUsedMemCurrent, size_t, __pool_attr_settable{false}>
{};

inline void __set_attribute_non_zero_only(::hipMemPool_t __pool, ::hipMemPoolAttr __attr, size_t __value)
{
  if (__value != 0)
  {
    _CCCL_THROW(::std::invalid_argument, "This attribute can't be set to a non-zero value.");
  }
  ::cuda::__driver::__mempoolSetAttribute(__pool, __attr, &__value);
}

template <>
struct __pool_attr<::hipMemPoolAttrReservedMemHigh>
    : __pool_attr_impl<::hipMemPoolAttrReservedMemHigh, size_t, __pool_attr_settable{true}>
{
  static void set(::hipMemPool_t __pool, type __value)
  {
    ::cuda::__set_attribute_non_zero_only(__pool, ::hipMemPoolAttrReservedMemHigh, __value);
  }
};

template <>
struct __pool_attr<::hipMemPoolAttrUsedMemHigh>
    : __pool_attr_impl<::hipMemPoolAttrUsedMemHigh, size_t, __pool_attr_settable{true}>
{
  static void set(::hipMemPool_t __pool, type __value)
  {
    ::cuda::__set_attribute_non_zero_only(__pool, ::hipMemPoolAttrUsedMemHigh, __value);
  }
};

namespace memory_pool_attributes
{
// The threshold at which the pool will release memory.
using release_threshold_t = __pool_attr<::hipMemPoolAttrReleaseThreshold>;
static constexpr release_threshold_t release_threshold{};

// Allow the pool to reuse the memory across streams as long as there is a
// stream ordering dependency between the streams.
using reuse_follow_event_dependencies_t = __pool_attr<::hipMemPoolReuseFollowEventDependencies>;
static constexpr reuse_follow_event_dependencies_t reuse_follow_event_dependencies{};

// Allow the pool to reuse already completed frees when there is no dependency
// between the streams.
using reuse_allow_opportunistic_t = __pool_attr<::hipMemPoolReuseAllowOpportunistic>;
static constexpr reuse_allow_opportunistic_t reuse_allow_opportunistic{};

// Allow the pool to insert stream dependencies to reuse the memory across
// streams.
using reuse_allow_internal_dependencies_t = __pool_attr<::hipMemPoolReuseAllowInternalDependencies>;
static constexpr reuse_allow_internal_dependencies_t reuse_allow_internal_dependencies{};

// The current amount of memory reserved in the pool.
using reserved_mem_current_t = __pool_attr<::hipMemPoolAttrReservedMemCurrent>;
static constexpr reserved_mem_current_t reserved_mem_current{};

// The high water mark for the reserved memory in the pool.
using reserved_mem_high_t = __pool_attr<::hipMemPoolAttrReservedMemHigh>;
static constexpr reserved_mem_high_t reserved_mem_high{};

// The current amount of memory used in the pool.
using used_mem_current_t = __pool_attr<::hipMemPoolAttrUsedMemCurrent>;
static constexpr used_mem_current_t used_mem_current{};

// The high water mark for the used memory in the pool.
using used_mem_high_t = __pool_attr<::hipMemPoolAttrUsedMemHigh>;
static constexpr used_mem_high_t used_mem_high{};
}; // namespace memory_pool_attributes

inline bool __is_host_memory_pool_supported()
{
  // Both host_numa and host memory pool flags should agree, but check the one corresponding to the implementation
  // of the default pool just to be sure
#  if _CCCL_CTK_AT_LEAST(13, 0)
  return ::cuda::device_attributes::host_memory_pools_supported(cuda::device_ref{0});
#  elif _CCCL_CTK_AT_LEAST(12, 9)
  return ::cuda::device_attributes::host_numa_memory_pools_supported(cuda::device_ref{0});
#  else
  return false;
#  endif
}

//! @brief  Checks whether the current device supports stream-ordered
//! allocations.
//! @param __device The device for which to query support.
//! @throws cuda_error if \c cudaDeviceGetAttribute failed.
//! @returns true if \c cudaDevAttrMemoryPoolsSupported is not zero.
inline void __verify_device_supports_stream_ordered_allocations(
  ::hipMemLocation __location, [[maybe_unused]] ::hipMemAllocationType __allocation_type)
{
  auto __device =
    __location.type == ::hipMemLocationTypeDevice ? cuda::device_ref{__location.id} : cuda::device_ref{0};
  if (!::cuda::device_attributes::memory_pools_supported(__device))
  {
    _CCCL_THROW(::cuda::cuda_error, ::hipErrorNotSupported, "stream-ordered allocations are not supported");
  }
#  if _CCCL_CTK_AT_LEAST(13, 0)
  if (__allocation_type == ::hipMemAllocationTypeMax
      && !::cuda::device_attributes::concurrent_managed_access(__device))
  {
    _CCCL_THROW(::cuda::cuda_error, ::hipErrorNotSupported, "managed memory pools are not supported");
  }
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
#  if _CCCL_CTK_AT_LEAST(12, 9)
  if (__location.type == ::hipMemLocationTypeHost && !__is_host_memory_pool_supported())
  {
    _CCCL_THROW(::cuda::cuda_error, ::hipErrorNotSupported, "host memory pools are not supported");
  }
#  endif // _CCCL_CTK_AT_LEAST(12, 9)
}

//! @brief Check whether the specified `cudaMemAllocationHandleType` is
//! supported on the present CUDA driver/runtime version.
//! @param __device The device to check for support.
//! @param __handle_type An IPC export handle type to check for support.
//! @throws cuda_error if the specified `cudaMemAllocationHandleType` is not
//! supported on the specified device.
inline void __verify_device_supports_export_handle_type(
  const device_ref __device, ::hipMemAllocationHandleType __handle_type, ::hipMemLocation __location)
{
  if (__handle_type == ::hipMemAllocationHandleType::hipMemHandleTypeNone)
  {
    return;
  }
  if (__location.type != ::hipMemLocationTypeDevice
#  if _CCCL_CTK_AT_LEAST(12, 9)
      && __location.type != ::hipMemLocationTypeHostNuma
#  endif
  )
  {
    _CCCL_THROW(::cuda::cuda_error,
                ::hipErrorNotSupported,
                "Requested IPC memory handle type not supported "
                "for the given location");
  }
  auto __supported_handles = __device.attribute(::cuda::device_attributes::memory_pool_supported_handle_types);
  if ((static_cast<int>(__handle_type) & __supported_handles) != static_cast<int>(__handle_type))
  {
    _CCCL_THROW(
      cuda::cuda_error, ::hipErrorNotSupported, "Requested IPC memory handle type not supported on a given device");
  }
}

[[nodiscard]] _CCCL_HOST_API inline hipMemPool_t
__get_default_memory_pool(const hipMemLocation __location, [[maybe_unused]] const hipMemAllocationType __allocation_type)
{
  ::cuda::__verify_device_supports_stream_ordered_allocations(__location, __allocation_type);

#  if _CCCL_CTK_AT_LEAST(13, 0)
  ::hipMemPool_t __pool = ::cuda::__driver::__getDefaultMemPool(__location, __allocation_type);
  if (::cuda::memory_pool_attributes::release_threshold(__pool) == 0)
  {
    ::cuda::memory_pool_attributes::release_threshold.set(__pool, ::cuda::std::numeric_limits<size_t>::max());
  }
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  _CCCL_ASSERT(__location.type == ::hipMemLocationTypeDevice,
               "Before CUDA 13 only device memory pools have a default");
  ::hipMemPool_t __pool = ::cuda::__driver::__deviceGetDefaultMemPool(::hipDevice_t{__location.id});
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
  return __pool;
}

//! @brief Enable access to this memory pool from the supplied devices
//!
//! Device on which this pool resides can be included in the span.
//!
//! @param __pool The memory pool to set access for
//! @param __devices A span of `device_ref`s listing devices to enable access
//! for
//! @param __flags The access flags to set
//! @throws cuda_error if ``cudaMemPoolSetAccess`` fails.
inline void
__mempool_set_access(::hipMemPool_t __pool, ::cuda::std::span<const device_ref> __devices, ::hipMemAccessFlags __flags)
{
  ::std::vector<::hipMemAccessDesc> __descs;
  __descs.reserve(__devices.size());
  for (const auto& __dev : __devices)
  {
    __descs.push_back({::hipMemLocation{::hipMemLocationTypeDevice, __dev.get()}, __flags});
  }
  ::cuda::__driver::__mempoolSetAccess(__pool, __descs.data(), __descs.size());
}

//! @brief Query if memory from a pool is accessible by the supplied device
//!
//! @param __pool The memory pool to query access for
//! @param __dev The device to query access for
//! @returns true if the memory pool is accessible from the device
[[nodiscard]] inline bool __mempool_get_access(::hipMemPool_t __pool, device_ref __dev)
{
  ::hipMemAccessFlags __result;
  ::hipMemLocation __loc;
  __loc.type = ::hipMemLocationTypeDevice;
  __loc.id   = __dev.get();
  __result   = ::cuda::__driver::__mempoolGetAccess(__pool, &__loc);
  return __result == ::hipMemAccessFlagsProtReadWrite;
}

//! @brief \c memory_pool_properties is a type that can controls memory pool to
//! control the creation options. Compared to attributes, properties can not be
//! set after the pool is created.
struct memory_pool_properties
{
  size_t initial_pool_size                           = 0;
  size_t release_threshold                           = ::cuda::std::numeric_limits<size_t>::max();
  hipMemAllocationHandleType allocation_handle_type = ::hipMemAllocationHandleType::hipMemHandleTypeNone;
  size_t max_pool_size                               = 0;
};

//! @brief  Creates the CUDA memory pool from the passed in arguments.
//! @throws cuda_error If the creation of the CUDA memory pool failed.
//! @returns The created CUDA memory pool.
[[nodiscard]] static hipMemPool_t __create_cuda_mempool(
  memory_pool_properties __properties, ::hipMemLocation __location, hipMemAllocationType __allocation_type)
{
  ::hipMemPoolProps __pool_properties{};
  __pool_properties.allocType   = __allocation_type;
  __pool_properties.handleTypes = ::hipMemAllocationHandleType(__properties.allocation_handle_type);
  __pool_properties.location    = __location;

#  if _CCCL_CTK_AT_LEAST(12, 2)
  if (__properties.max_pool_size != 0)
  {
#    if _CCCL_CTK_AT_LEAST(13, 0)
    if (__allocation_type == ::hipMemAllocationTypeMax)
    {
      _CCCL_THROW(::std::invalid_argument, "Max pool size is not supported for managed memory pools");
    }
#    endif // _CCCL_CTK_AT_LEAST(13, 0)
    if (__properties.initial_pool_size > __properties.max_pool_size)
    {
      _CCCL_THROW(::std::invalid_argument, "Initial pool size must be less than the max pool size");
    }
  }
  __pool_properties.maxSize = __properties.max_pool_size;
#  else
  if (__properties.max_pool_size != 0)
  {
    _CCCL_THROW(::std::invalid_argument, "Max pool size is not supported on this CUDA version");
  }
#  endif // _CCCL_CTK_AT_LEAST(12, 2)

  if (__properties.initial_pool_size > __properties.release_threshold)
  {
    _CCCL_THROW(::std::invalid_argument, "Initial pool size must be less than the release threshold");
  }

  ::hipMemPool_t __cuda_pool_handle{};
  ::hipError_t __error = ::cuda::__driver::__mempoolCreateNoThrow(&__cuda_pool_handle, &__pool_properties);
  if (__error != ::hipSuccess)
  {
    auto __device = __location.type == ::hipMemLocationTypeDevice ? __location.id : 0;
    // Mempool creation failed, lets try to figure out why
    ::cuda::__verify_device_supports_stream_ordered_allocations(__location, __allocation_type);
    ::cuda::__verify_device_supports_export_handle_type(__device, __properties.allocation_handle_type, __location);

    // Could not find the reason, throw a generic error
    _CCCL_THROW(::cuda::cuda_error, __error, "Failed to create a memory pool");
  }

  ::cuda::__driver::__mempoolSetAttribute(
    __cuda_pool_handle, ::hipMemPoolAttrReleaseThreshold, &__properties.release_threshold);

  // allocate the requested initial size to prime the pool.
  // We need to use a new stream so we do not wait on other work
  if (__properties.initial_pool_size != 0)
  {
    ::hipDeviceptr_t __ptr = ::cuda::__driver::__mallocFromPoolAsync(
      __properties.initial_pool_size, __cuda_pool_handle, __cccl_allocation_stream().get());
    if (::cuda::__driver::__freeAsyncNoThrow(__ptr, __cccl_allocation_stream().get()) != ::hipSuccess)
    {
      _CCCL_THROW(::cuda::cuda_error, ::hipErrorMemoryAllocation, "Failed to allocate initial pool size");
    }
  }
  return __cuda_pool_handle;
}

class __memory_pool_base
{
protected:
  ::hipMemPool_t __pool_;

  //! @brief Checks whether the passed in alignment is valid.
  //! @param __alignment the alignment to check.
  //! @returns true if \p __alignment is valid.
  [[nodiscard]] _CCCL_HOST_API static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= ::cuda::mr::default_cuda_malloc_alignment
        && (::cuda::mr::default_cuda_malloc_alignment % __alignment == 0);
  }

public:
  __memory_pool_base(::cuda::std::nullptr_t) = delete;

  //! @brief  Constructs the __memory_pool_base from a \c cudaMemPool_t.
  //! @param __pool The \c cudaMemPool_t used to allocate memory.
  _CCCL_HOST_API explicit __memory_pool_base(::hipMemPool_t __pool) noexcept
      : __pool_(__pool)
  {}

  //! @brief Allocate device memory of size at least \p __bytes via
  //! cudaMallocFromPoolAsync.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @throws std::invalid_argument In case of invalid alignment.
  //! @throws cuda::cuda_error If an error code was return by the CUDA API call.
  //! @returns Pointer to the newly allocated memory.
  [[nodiscard]] _CCCL_HOST_API void*
  allocate_sync(const size_t __bytes, const size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment)
  {
    if (!__is_valid_alignment(__alignment))
    {
      _CCCL_THROW(::std::invalid_argument, "Invalid alignment passed to __memory_pool_base::allocate_sync.");
    }

    ::hipDeviceptr_t __ptr = ::cuda::__driver::__mallocFromPoolAsync(__bytes, __pool_, __cccl_allocation_stream().get());
    __cccl_allocation_stream().sync();
    return reinterpret_cast<void*>(__ptr); // NOLINT(performance-no-int-to-ptr)
  }

  //! @brief deallocate_sync memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a
  //! call to `allocate_sync`.
  //! @param __bytes  The number of bytes that was passed to the allocation call
  //! that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the allocation call
  //! that returned \p __ptr.
  //! @note The pointer passed to `deallocate_sync` must not be in use in a
  //! stream. It is the caller's responsibility to properly synchronize all
  //! relevant streams before calling `deallocate_sync`.
  _CCCL_HOST_API void deallocate_sync(
    void* __ptr,
    const size_t,
    [[maybe_unused]] const size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment) noexcept
  {
    _CCCL_ASSERT(__is_valid_alignment(__alignment), "Invalid alignment passed to __memory_pool_base::deallocate_sync.");
    _CCCL_ASSERT_CUDA_API(
      ::cuda::__driver::__freeAsyncNoThrow,
      "deallocate failed",
      reinterpret_cast<::hipDeviceptr_t>(__ptr),
      __cccl_allocation_stream().get());
  }

  //! @brief Allocate device memory of size at least \p __bytes via
  //! `cudaMallocFromPoolAsync`.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @param __stream Stream on which to perform allocation.
  //! @throws std::invalid_argument In case of invalid alignment.
  //! @throws cuda::cuda_error If an error code was return by the cuda api call.
  //! @returns Pointer to the newly allocated memory.
  [[nodiscard]] _CCCL_HOST_API void*
  allocate(const ::cuda::stream_ref __stream, const size_t __bytes, const size_t __alignment)
  {
    if (!__is_valid_alignment(__alignment))
    {
      _CCCL_THROW(::std::invalid_argument, "Invalid alignment passed to __memory_pool_base::allocate.");
    }

    return allocate(__stream, __bytes);
  }

  //! @brief Allocate device memory of size at least \p __bytes via
  //! cudaMallocFromPoolAsync.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __stream Stream on which to perform allocation.
  //! @throws cuda::cuda_error If an error code was return by the cuda api call.
  //! @returns Pointer to the newly allocated memory.
  [[nodiscard]] _CCCL_HOST_API void* allocate(const ::cuda::stream_ref __stream, const size_t __bytes)
  {
    ::hipDeviceptr_t __ptr = ::cuda::__driver::__mallocFromPoolAsync(__bytes, __pool_, __stream.get());
    return reinterpret_cast<void*>(__ptr); // NOLINT(performance-no-int-to-ptr)
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a
  //! call to `allocate`
  //! @param __bytes The number of bytes that was passed to the allocation call
  //! that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the allocation call
  //! that returned \p __ptr.
  //! @param __stream A stream that has a stream ordering relationship with the
  //! stream used in the <a
  //! href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">allocate</a>
  //! call that returned \p __ptr.
  //! @note The pointer passed to `deallocate` must not be in use in a stream
  //! other than \p __stream. It is the caller's responsibility to properly
  //! synchronize all relevant streams before calling `deallocate`.
  void _CCCL_HOST_API deallocate(
    const ::cuda::stream_ref __stream,
    void* __ptr,
    const size_t __bytes,
    [[maybe_unused]] const size_t __alignment) noexcept
  {
    // We need to ensure that the provided alignment matches the minimal
    // provided alignment
    _CCCL_ASSERT(__is_valid_alignment(__alignment), "Invalid alignment passed to __memory_pool_base::deallocate.");
    deallocate(__stream, __ptr, __bytes);
  }

  //! @brief Tries to release memory.
  //! @param __min_bytes_to_keep the minimal guaranteed size of the pool.
  //! @note If the pool has less than \p __minBytesToKeep reserved, the trim_to
  //! operation is a no-op. Otherwise the pool will be guaranteed to have at
  //! least \p __minBytesToKeep bytes reserved after the operation.
  _CCCL_HOST_API void trim_to(const size_t __min_bytes_to_keep)
  {
    ::cuda::__driver::__mempoolTrimTo(__pool_, __min_bytes_to_keep);
  }

  //! @brief Gets the value of an attribute of the pool.
  //! @param __attr the attribute to get.
  //! @return The value of the attribute.
  template <typename _Attr>
  [[nodiscard]] _CCCL_HOST_API auto attribute(_Attr __attr) const
  {
    return __attr(__pool_);
  }

  //! @brief Gets the value of an attribute of the pool.
  //! @param __attribute the attribute to get.
  //! @return The value of the attribute.
  template <::hipMemPoolAttr _Attr>
  _CCCL_HOST_API auto attribute() const
  {
    return attribute(__pool_attr<_Attr>());
  }

  //! @brief Sets an attribute of the pool to a given value.
  //! @param __attribute the attribute to be set.
  //! @param __value the new value of that attribute.
  template <typename _Attr>
  _CCCL_HOST_API void set_attribute(_Attr __attr, typename _Attr::type __value)
  {
    __attr.set(__pool_, __value);
  }

  //! @brief Sets an attribute of the pool to a given value.
  //! @param __attribute the attribute to be set.
  //! @param __value the new value of that attribute.
  template <::hipMemPoolAttr _Attr>
  _CCCL_HOST_API void set_attribute(typename __pool_attr<_Attr>::type __value)
  {
    return set_attribute(__pool_attr<_Attr>(), __value);
  }

  //! @brief Returns the underlying handle to the CUDA memory pool.
  [[nodiscard]] _CCCL_HOST_API constexpr hipMemPool_t get() const noexcept
  {
    return __pool_;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a
  //! call to `allocate`.
  //! @param __bytes The number of bytes that was passed to the allocation call
  //! that returned \p __ptr.
  //! @param __stream A stream that has a stream ordering relationship with the
  //! stream used in the <a
  //! href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">allocate</a>
  //! call that returned \p __ptr.
  //! @note The pointer passed to `deallocate` must not be in use in a stream
  //! other than \p __stream. It is the caller's responsibility to properly
  //! synchronize all relevant streams before calling `deallocate`.
  _CCCL_HOST_API void deallocate(const ::cuda::stream_ref __stream, void* __ptr, size_t) noexcept
  {
    _CCCL_ASSERT_CUDA_API(
      ::cuda::__driver::__freeAsyncNoThrow, "deallocate failed", reinterpret_cast<::hipDeviceptr_t>(__ptr), __stream.get());
  }

  //! @brief Enable access to memory allocated through this memory resource by
  //! the supplied devices
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //! Device on which this resource allocates memory can be included in the
  //! span.
  //!
  //! @param __devices A span of `device_ref`s listing devices to enable access
  //! for
  _CCCL_HOST_API void enable_access_from(::cuda::std::span<const device_ref> __devices)
  {
    ::cuda::__mempool_set_access(__pool_, {__devices.data(), __devices.size()}, ::hipMemAccessFlagsProtReadWrite);
  }

  //! @brief Enable access to memory allocated through this memory resource by
  //! the supplied device
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //!
  //! @param __device device_ref indicating for which device the access should
  //! be enabled
  _CCCL_HOST_API void enable_access_from(device_ref __device)
  {
    ::cuda::__mempool_set_access(__pool_, {&__device, 1}, ::hipMemAccessFlagsProtReadWrite);
  }

  //! @brief Disable access to memory allocated through this memory resource by
  //! the supplied devices
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //! Device on which this resource allocates memory can be included in the
  //! span.
  //!
  //! @param __devices A span of `device_ref`s listing devices to disable access
  //! for
  _CCCL_HOST_API void disable_access_from(::cuda::std::span<const device_ref> __devices)
  {
    ::cuda::__mempool_set_access(__pool_, {__devices.data(), __devices.size()}, ::hipMemAccessFlagsProtNone);
  }

  //! @brief Disable access to memory allocated through this memory resource by
  //! the supplied device
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //!
  //! @param __device device_ref indicating for which device the access should
  //! be disabled
  _CCCL_HOST_API void disable_access_from(device_ref __device)
  {
    ::cuda::__mempool_set_access(__pool_, {&__device, 1}, ::hipMemAccessFlagsProtNone);
  }

  //! @brief Query if memory allocated through this memory resource is
  //! accessible by the supplied device
  //!
  //! @param __device device for which the access is queried
  [[nodiscard]] _CCCL_HOST_API bool is_accessible_from(device_ref __device)
  {
    return ::cuda::__mempool_get_access(__pool_, __device);
  }

  //! @brief Equality comparison with another __memory_pool_base.
  //! @returns true if underlying \c cudaMemPool_t are equal.
  [[nodiscard]] _CCCL_HOST_API bool operator==(__memory_pool_base const& __rhs) const noexcept
  {
    return __pool_ == __rhs.__pool_;
  }

#  if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison with another __memory_pool_base.
  //! @returns true if underlying \c cudaMemPool_t are not equal.
  [[nodiscard]] _CCCL_HOST_API bool operator!=(__memory_pool_base const& __rhs) const noexcept
  {
    return __pool_ != __rhs.__pool_;
  }
#  endif // _CCCL_STD_VER <= 2017
};

_CCCL_END_NAMESPACE_CUDA

_CCCL_DIAG_POP

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#endif // _CUDA___MEMORY_RESOURCE_MEMORY_POOL_BASE_H
