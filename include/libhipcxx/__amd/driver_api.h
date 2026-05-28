// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

// NOTE(HIP/AMD): HIP-side replacement for the libhipcxx CUDA driver-API
// helpers in `<cuda/__driver/driver_api.h>`. Provides the subset of
// `::cuda::__driver::__xxx` wrappers that the upstream code consumes,
// implemented directly on top of the HIP runtime / driver API.
//
// Each wrapper below picks the HIP API that best matches the *semantics*
// the CUDA-side wrapper provides, NOT necessarily the closest function
// name. In particular, the pointer-attribute helpers route through
// `hipPointerGetAttributes` (plural runtime API) rather than
// `hipPointerGetAttribute` (singular driver API), because the plural
// variant matches the original CUDA-runtime semantics expected by the
// upstream callers (e.g., it gracefully reports unregistered host memory
// instead of returning `hipErrorInvalidValue`).

#ifndef _AMD_DRIVER_API_H
#define _AMD_DRIVER_API_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HIP_COMPILATION() && !defined(_CCCL_COMPILER_HIPRTC)

#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__internal/namespaces.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_same.h>

#  include <cstddef> // for std::size_t in the mempool / managed-alloc helpers

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DRIVER

// Device management

[[nodiscard]] _CCCL_HOST_API inline int __cudevice_to_ordinal(::hipDevice_t __dev) noexcept
{
  // hipDevice_t is a plain int representing the ordinal.
  return static_cast<int>(__dev);
}

[[nodiscard]] _CCCL_HOST_API inline ::hipDevice_t __deviceGet(int __ordinal)
{
  ::hipDevice_t __result{};
  _CCCL_TRY_CUDA_API(::hipDeviceGet, "Failed to get device", &__result, __ordinal);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::hipDevice_t
__deviceGetAttribute(::hipDeviceAttribute_t __attr, ::hipDevice_t __device)
{
  int __result = 0;
  _CCCL_TRY_CUDA_API(::hipDeviceGetAttribute, "Failed to get device attribute", &__result, __attr, __device);
  return static_cast<::hipDevice_t>(__result);
}

[[nodiscard]] _CCCL_HOST_API inline int __deviceGetCount()
{
  int __result = 0;
  _CCCL_TRY_CUDA_API(::hipGetDeviceCount, "Failed to get device count", &__result);
  return __result;
}

_CCCL_HOST_API inline void __deviceGetName(char* __name_out, int __len, int __ordinal)
{
  ::hipDevice_t __dev = __deviceGet(__ordinal);
  _CCCL_TRY_CUDA_API(::hipDeviceGetName, "Failed to query the name of a device", __name_out, __len, __dev);
}

[[nodiscard]] _CCCL_HOST_API inline bool __deviceCanAccessPeer(::hipDevice_t __dev, ::hipDevice_t __peer_dev)
{
  int __result = 0;
  _CCCL_TRY_CUDA_API(::hipDeviceCanAccessPeer, "Failed to query peer access", &__result, __dev, __peer_dev);
  return __result != 0;
}

// NOTE(HIP/AMD): the CUDA driver-API context-stack model (cuCtxPush/Pop,
// cuCtxGetCurrent, cuDevicePrimaryCtx*) has no faithful HIP equivalent --
// the HIP runtime model is "current device per thread" via
// hipSetDevice/hipGetDevice with no explicit ctx stack and no refcounted
// primary-context objects. The corresponding cuda::__driver::__primaryCtx*
// / __ctxPush/Pop / __ctxGetCurrent / __ctxGetDevice / __isPrimaryCtxActive
// shims are therefore deliberately NOT provided. Consumers that previously
// went through those shims (__ensure_current_context, __physical_device,
// stream_ref) take a HIP-specific code path that calls hipSetDevice /
// hipGetDevice directly.

// NOTE(HIP/AMD): mirrors upstream cuda::__driver::__getVersion(). HIP
// returns the driver version as the runtime version on
// hipDriverGetVersion (semantically same as CUDA's cuDriverGetVersion).
[[nodiscard]] _CCCL_HOST_API inline int __getVersion()
{
  int __result{};
  _CCCL_TRY_CUDA_API(::hipDriverGetVersion, "Failed to get driver version", &__result);
  return __result;
}

// Pointer attributes
//
// Routed through `hipPointerGetAttributes` (plural runtime API) so that
// unregistered host pointers (stack/heap arrays) are reported gracefully
// instead of returning `hipErrorInvalidValue` like the singular driver-API
// `hipPointerGetAttribute` would. The translation below maps the HIP
// runtime types onto the CUmemorytype values the upstream callers expect
// (the AMD shim aliases `CUmemorytype` to `hipMemoryType`, so the values
// coincide modulo the `Unregistered` case which we collapse to `Host`).

template <::hipPointer_attribute _Attr>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __pointer_attribute_value_type_t_impl() noexcept
{
  if constexpr (_Attr == ::HIP_POINTER_ATTRIBUTE_CONTEXT)
  {
    return ::hipCtx_t{};
  }
  else if constexpr (_Attr == ::HIP_POINTER_ATTRIBUTE_MEMORY_TYPE)
  {
    return ::hipMemoryType{};
  }
  else if constexpr (_Attr == ::HIP_POINTER_ATTRIBUTE_DEVICE_POINTER || _Attr == ::HIP_POINTER_ATTRIBUTE_HOST_POINTER)
  {
    return static_cast<void*>(nullptr);
  }
  else if constexpr (_Attr == ::HIP_POINTER_ATTRIBUTE_IS_MANAGED || _Attr == ::HIP_POINTER_ATTRIBUTE_MAPPED)
  {
    return bool{};
  }
  else if constexpr (_Attr == ::HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL)
  {
    return int{};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<decltype(_Attr)>, "not implemented attribute");
  }
}

template <::hipPointer_attribute _Attr>
using __pointer_attribute_value_type_t = decltype(::cuda::__driver::__pointer_attribute_value_type_t_impl<_Attr>());

// Full-struct query: a single shim that wraps `hipPointerGetAttributes`
// and returns the entire `hipPointerAttribute_t` to the caller. This is
// the natural HIP equivalent of CUDA's variadic
// `cuPointerGetAttributes(...)`/`__pointerGetAttributesNoThrow<_Np>`
// path -- the HIP runtime always returns every field in one call, so
// callers that need more than one attribute (e.g. type + device +
// isManaged) can avoid duplicating the runtime call. Callers that only
// need a single attribute should keep using the per-attribute templated
// `__pointerGetAttributeNoThrow<_Attr>` helper just below, which adds
// the CUDA-side semantic massaging (e.g. collapsing `Unregistered` to
// `Host` for MEMORY_TYPE).
//
// On older ROCm releases this returns `hipErrorInvalidValue` for
// unregistered host memory (stack/heap/static), so callers must accept
// that as a non-fatal "unknown pointer" outcome -- matching what they
// already do for `cudaErrorInvalidValue` on CUDA.
[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__pointerGetAttributesNoThrow(::hipPointerAttribute_t& __result, const void* __ptr) noexcept
{
  // Zero-initialise so an early-out (hipErrorInvalidValue) leaves the
  // type field as hipMemoryTypeUnregistered (= 0) for callers that
  // choose to inspect __result regardless of the returned status.
  __result = {};
  return static_cast<::cudaError_t>(::hipPointerGetAttributes(&__result, __ptr));
}

template <::hipPointer_attribute _Attr>
[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__pointerGetAttributeNoThrow(__pointer_attribute_value_type_t<_Attr>& __result, const void* __ptr)
{
  ::hipPointerAttribute_t __ptr_attrib{};
  const auto __status = ::cuda::__driver::__pointerGetAttributesNoThrow(__ptr_attrib, __ptr);
  if (__status != ::cudaSuccess)
  {
    return __status;
  }
  if constexpr (_Attr == ::HIP_POINTER_ATTRIBUTE_MEMORY_TYPE)
  {
    // Treat unregistered host memory as host so callers get the same
    // semantics as `cuPointerGetAttribute(MEMORY_TYPE)` on CUDA.
    __result =
      (__ptr_attrib.type == ::hipMemoryTypeUnregistered) ? ::hipMemoryTypeHost : __ptr_attrib.type;
  }
  else if constexpr (_Attr == ::HIP_POINTER_ATTRIBUTE_IS_MANAGED)
  {
    // NOTE(HIP/AMD): tracked as AIRUNTIME-2114. The HIP runtime does not
    // currently report `__managed__` global variables as managed. Use the
    // "devicePointer == hostPointer" heuristic that works for explicitly
    // `hipMallocManaged()`-allocated memory.
    __result = (__ptr_attrib.devicePointer != nullptr)
            && (__ptr_attrib.hostPointer == __ptr_attrib.devicePointer);
  }
  else if constexpr (_Attr == ::HIP_POINTER_ATTRIBUTE_DEVICE_POINTER)
  {
    __result = __ptr_attrib.devicePointer;
  }
  else if constexpr (_Attr == ::HIP_POINTER_ATTRIBUTE_HOST_POINTER)
  {
    __result = __ptr_attrib.hostPointer;
  }
  else if constexpr (_Attr == ::HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL)
  {
    __result = __ptr_attrib.device;
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<decltype(_Attr)>, "not implemented attribute");
  }
  return ::hipSuccess;
}

// Stream management

[[nodiscard]] _CCCL_HOST_API inline ::hipStream_t __streamCreateWithPriority(unsigned __flags, int __priority)
{
  ::hipStream_t __stream{};
  _CCCL_TRY_CUDA_API(::hipStreamCreateWithPriority, "Failed to create a stream", &__stream, __flags, __priority);
  return __stream;
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __streamDestroyNoThrow(::hipStream_t __stream)
{
  return static_cast<::cudaError_t>(::hipStreamDestroy(__stream));
}

_CCCL_HOST_API inline void __streamSynchronize(::hipStream_t __stream)
{
  _CCCL_TRY_CUDA_API(::hipStreamSynchronize, "Failed to synchronize a stream", __stream);
}

[[nodiscard]] _CCCL_HOST_API inline ::hipDevice_t __streamGetDevice(::hipStream_t __stream)
{
  ::hipDevice_t __result{};
  _CCCL_TRY_CUDA_API(::hipStreamGetDevice, "Failed to get the device of a stream", __stream, &__result);
  return __result;
}

_CCCL_HOST_API inline void __streamWaitEvent(::hipStream_t __stream, ::hipEvent_t __evnt)
{
  _CCCL_TRY_CUDA_API(::hipStreamWaitEvent, "Failed to make a stream wait for an event", __stream, __evnt, 0u);
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __streamQueryNoThrow(::hipStream_t __stream)
{
  return static_cast<::cudaError_t>(::hipStreamQuery(__stream));
}

[[nodiscard]] _CCCL_HOST_API inline int __streamGetPriority(::hipStream_t __stream)
{
  int __result = 0;
  _CCCL_TRY_CUDA_API(::hipStreamGetPriority, "Failed to get priority of a stream", __stream, &__result);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline unsigned long long __streamGetId(::hipStream_t __stream)
{
  unsigned long long __result = 0;
  _CCCL_TRY_CUDA_API(::hipStreamGetId, "Failed to get ID of a stream", __stream, &__result);
  return __result;
}

// NOTE(HIP/AMD): `__streamGetCtx` is intentionally not provided -
// `hipStreamGetCtx` is not in the HIP runtime API. Consumers (currently
// `cuda::stream_ref::device()` and `__ensure_current_context(stream_ref)`)
// route through `__streamGetDevice` under HIP instead.

// Memory management
//
// NOTE(HIP/AMD): the upstream wrappers call `cuMemcpyAsync` /
// `cuMemsetD8Async` (driver-API). The HIP runtime equivalents are
// `hipMemcpyAsync` (which requires an explicit `hipMemcpyKind` - we
// pass `hipMemcpyDefault` to match the upstream UVA-based semantics)
// and `hipMemsetD8Async` (which takes a `hipDeviceptr_t`, an alias of
// `void*` on HIP).

_CCCL_HOST_API inline void __memcpyAsync(void* __dst, const void* __src, size_t __count, ::hipStream_t __stream)
{
  _CCCL_TRY_CUDA_API(
    ::hipMemcpyAsync, "Failed to perform a memcpy", __dst, __src, __count, ::hipMemcpyDefault, __stream);
}

_CCCL_HOST_API inline void __memsetAsync(void* __dst, ::uint8_t __value, size_t __count, ::hipStream_t __stream)
{
  _CCCL_TRY_CUDA_API(
    ::hipMemsetD8Async,
    "Failed to perform a memset",
    reinterpret_cast<::hipDeviceptr_t>(__dst),
    __value,
    __count,
    __stream);
}

// Event management

_CCCL_HOST_API inline void __eventRecord(::hipEvent_t __evnt, ::hipStream_t __stream)
{
  _CCCL_TRY_CUDA_API(::hipEventRecord, "Failed to record CUDA event", __evnt, __stream);
}

_CCCL_HOST_API inline void __eventSynchronize(::hipEvent_t __evnt)
{
  _CCCL_TRY_CUDA_API(::hipEventSynchronize, "Failed to synchronize CUDA event", __evnt);
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __eventQueryNoThrow(::hipEvent_t __evnt)
{
  return static_cast<::cudaError_t>(::hipEventQuery(__evnt));
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __eventDestroyNoThrow(::hipEvent_t __evnt)
{
  return static_cast<::cudaError_t>(::hipEventDestroy(__evnt));
}

[[nodiscard]] _CCCL_HOST_API inline ::hipEvent_t __eventCreate(unsigned __flags)
{
  ::hipEvent_t __result{};
  _CCCL_TRY_CUDA_API(::hipEventCreateWithFlags, "Failed to create CUDA event", &__result, __flags);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline float __eventElapsedTime(::hipEvent_t __start, ::hipEvent_t __end)
{
  float __result = 0.0f;
  _CCCL_TRY_CUDA_API(
    ::hipEventElapsedTime, "Failed to get elapsed time between CUDA events", &__result, __start, __end);
  return __result;
}

// NOTE(HIP/AMD): stream-callback / host-launch wrappers used by
// <cuda/__launch/host_launch.h>. The CUDA driver-API form takes
// ::CUstreamCallback / ::CUhostFn; HIP equivalents are
// ::hipStreamCallback_t / ::hipHostFn_t.
_CCCL_HOST_API inline void
__streamAddCallback(::hipStream_t __stream, ::hipStreamCallback_t __cb, void* __data, unsigned __flags = 0)
{
  _CCCL_TRY_CUDA_API(
    ::hipStreamAddCallback, "Failed to add a stream callback", __stream, __cb, __data, __flags);
}

_CCCL_HOST_API inline void __launchHostFunc(::hipStream_t __stream, ::hipHostFn_t __fn, void* __data)
{
  _CCCL_TRY_CUDA_API(::hipLaunchHostFunc, "Failed to launch host function", __stream, __fn, __data);
}

// NOTE(HIP/AMD): memory-pool attribute getter/setter used by
// <cuda/__memory_resource/memory_resource_base.h>. The CUDA form
// takes ::CUmemPool_attribute which we alias to ::hipMemPoolAttr in
// <libhipcxx/__amd/cuda_runtime.h>.
[[nodiscard]] _CCCL_HOST_API inline size_t
__mempoolGetAttribute(::hipMemPool_t __pool, ::hipMemPoolAttr __attr)
{
  size_t __value = 0;
  _CCCL_TRY_CUDA_API(
    ::hipMemPoolGetAttribute, "Failed to get attribute for a memory pool", __pool, __attr, &__value);
  return __value;
}

_CCCL_HOST_API inline void
__mempoolSetAttribute(::hipMemPool_t __pool, ::hipMemPoolAttr __attr, void* __value)
{
  _CCCL_TRY_CUDA_API(
    ::hipMemPoolSetAttribute, "Failed to set attribute for a memory pool", __pool, __attr, __value);
}

// NOTE(HIP/AMD): memory-pool create / access wrappers used by
// <cuda/__memory_resource/memory_resource_base.h>. The CUDA driver API
// uses CUmemoryPool/CUmemPoolProps/CUmemAccessDesc/CUmemAccess_flags
// which we alias to hipMemPool_t/hipMemPoolProps/hipMemAccessDesc/
// hipMemAccessFlags in <libhipcxx/__amd/cuda_runtime.h>. The HIP runtime API
// signature matches the upstream cuda::__driver shape exactly for
// these four entry points.
[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__mempoolCreateNoThrow(::hipMemPool_t* __pool, ::hipMemPoolProps* __props)
{
  return static_cast<::cudaError_t>(::hipMemPoolCreate(__pool, __props));
}

_CCCL_HOST_API inline void
__mempoolSetAccess(::hipMemPool_t __pool, ::hipMemAccessDesc* __descs, ::size_t __count)
{
  _CCCL_TRY_CUDA_API(::hipMemPoolSetAccess, "Failed to set access of a memory pool", __pool, __descs, __count);
}

[[nodiscard]] _CCCL_HOST_API inline ::hipMemAccessFlags
__mempoolGetAccess(::hipMemPool_t __pool, ::hipMemLocation* __location)
{
  ::hipMemAccessFlags __flags{};
  _CCCL_TRY_CUDA_API(::hipMemPoolGetAccess, "Failed to get access of a memory pool", &__flags, __pool, __location);
  return __flags;
}

// NOTE(HIP/AMD): allocation-from-pool wrappers used by the upstream
// __memory_resource_base. The CUDA driver API
//   ::cuda::__driver::__mallocFromPoolAsync(size, pool, stream)
//     -> ::CUdeviceptr
// returns the device pointer. HIP's hipMallocFromPoolAsync writes the
// pointer through an out-parameter and returns hipError_t -- adapt the
// signature here. The upstream consumer code keeps using the
// CUdeviceptr-returning shape.
[[nodiscard]] _CCCL_HOST_API inline ::hipDeviceptr_t
__mallocFromPoolAsync(::size_t __bytes, ::hipMemPool_t __pool, ::hipStream_t __stream)
{
  void* __result = nullptr;
  _CCCL_TRY_CUDA_API(
    ::hipMallocFromPoolAsync, "Failed to allocate memory from a memory pool", &__result, __bytes, __pool, __stream);
  return static_cast<::hipDeviceptr_t>(__result);
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__freeAsyncNoThrow(::hipDeviceptr_t __dptr, ::hipStream_t __stream)
{
  return static_cast<::cudaError_t>(::hipFreeAsync(__dptr, __stream));
}

_CCCL_HOST_API inline void __mempoolDestroy(::hipMemPool_t __pool)
{
  _CCCL_TRY_CUDA_API(::hipMemPoolDestroy, "Failed to destroy a memory pool", __pool);
}

_CCCL_HOST_API inline void __mempoolTrimTo(::hipMemPool_t __pool, ::size_t __min_bytes_to_keep)
{
  _CCCL_TRY_CUDA_API(::hipMemPoolTrimTo, "Failed to trim a memory pool", __pool, __min_bytes_to_keep);
}

// NOTE(HIP/AMD): managed/host allocator wrappers used by the legacy
// memory_resource implementations. The HIP runtime API
// (hipMallocManaged / hipHostMalloc / hipFree / hipFreeHost) writes
// the pointer through an out-parameter and returns hipError_t; adapt
// to match the upstream cuda::__driver shape (returns the pointer).
[[nodiscard]] _CCCL_HOST_API inline ::hipDeviceptr_t __mallocManaged(::size_t __bytes, unsigned int __flags)
{
  void* __result = nullptr;
  _CCCL_TRY_CUDA_API(::hipMallocManaged, "Failed to allocate managed memory", &__result, __bytes, __flags);
  return static_cast<::hipDeviceptr_t>(__result);
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __freeNoThrow(::hipDeviceptr_t __dptr)
{
  return static_cast<::cudaError_t>(::hipFree(__dptr));
}

[[nodiscard]] _CCCL_HOST_API inline void* __mallocHost(::size_t __bytes)
{
  void* __result = nullptr;
  // NOTE(HIP/AMD): hipHostMalloc takes a flags parameter (defaulting to
  // hipHostMallocDefault). Match the upstream zero-flags semantics by
  // explicitly passing 0u.
  _CCCL_TRY_CUDA_API(::hipHostMalloc, "Failed to allocate host memory", &__result, __bytes, 0u);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __freeHostNoThrow(void* __dptr)
{
  return static_cast<::cudaError_t>(::hipHostFree(__dptr));
}

_CCCL_END_NAMESPACE_CUDA_DRIVER

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HIP_COMPILATION() && !defined(_CCCL_COMPILER_HIPRTC)

#endif // _AMD_DRIVER_API_H
