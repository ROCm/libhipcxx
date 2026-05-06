//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
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

#ifndef _CUDA___MDSPAN_HOST_DEVICE_ACCESSOR_H
#define _CUDA___MDSPAN_HOST_DEVICE_ACCESSOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__driver/driver_api.h>
#include <cuda/__memory/address_space.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Accessor>
class __host_accessor;

template <typename _Accessor>
class __device_accessor;

template <typename _Accessor>
class __managed_accessor;

template <typename _Accessor>
using host_accessor = __host_accessor<_Accessor>;

template <typename _Accessor>
using device_accessor = __device_accessor<_Accessor>;

template <typename _Accessor>
using managed_accessor = __managed_accessor<_Accessor>;

/***********************************************************************************************************************
 * Host/Device/Managed Accessor Traits
 **********************************************************************************************************************/

template <typename>
inline constexpr bool is_host_accessor_v = false;

template <typename>
inline constexpr bool is_device_accessor_v = false;

template <typename>
inline constexpr bool is_managed_accessor_v = false;

template <typename _Accessor>
inline constexpr bool is_host_accessor_v<__host_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_device_accessor_v<__device_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_managed_accessor_v<__managed_accessor<_Accessor>> = true;

template <typename _Tp>
inline constexpr bool is_host_device_managed_accessor_v =
  is_host_accessor_v<_Tp> || is_device_accessor_v<_Tp> || is_managed_accessor_v<_Tp>;

/***********************************************************************************************************************
 * Host Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
class __host_accessor : public _Accessor
{
  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::__host_accessor/cuda::__device_accessor/cuda::__managed_accessor cannot be nested");

  using __data_handle_type = typename _Accessor::data_handle_type;

  static constexpr bool __is_access_noexcept =
    noexcept(::cuda::std::declval<_Accessor>().access(::cuda::std::declval<__data_handle_type>(), 0));

  static constexpr bool __is_offset_noexcept =
    noexcept(::cuda::std::declval<_Accessor>().offset(::cuda::std::declval<__data_handle_type>(), 0));

#if !_CCCL_COMPILER(NVRTC)
  [[nodiscard]] _CCCL_HOST_API static constexpr bool
  __is_host_accessible_pointer([[maybe_unused]] __data_handle_type __p) noexcept
  {
#  if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()
    if constexpr (::cuda::std::contiguous_iterator<__data_handle_type>)
    {
      if (!cuda::std::__cccl_default_is_constant_evaluated())
      {
        auto __p1 = ::cuda::std::to_address(__p);
#    if _CCCL_HIP_COMPILATION()
        // NOTE(HIP/AMD): use the cudaPointerGetAttributes runtime API
        // (shimmed to hipPointerGetAttributes via include/amd/cuda_runtime.h)
        // since the cuPointerGetAttribute driver API is not exposed by HIP
        // through cuda::__driver::__pointerGetAttributeNoThrow. Mirrors the
        // pattern used in __is_managed_pointer below.
        //
        // The Unregistered case is included because cudaPointerGetAttributes
        // returns 'cudaMemoryTypeUnregistered' for any pointer the runtime
        // does not know about (stack pointers, statically-allocated globals,
        // anything from regular new / malloc, ...) -- ALL of which ARE
        // host-accessible. The CUDA-side path on the right achieves the same
        // effect via 'status != cudaSuccess' (when the driver does not
        // recognise the pointer, fall through to "host"); we get there
        // explicitly via the type enum instead. Managed memory is also
        // host-accessible (unified addressing), hence the third clause.
        ::cudaPointerAttributes __ptr_attrib{};
        _CCCL_ASSERT_CUDA_API(::cudaPointerGetAttributes, "cudaPointerGetAttributes failed", &__ptr_attrib, __p1);
        return __ptr_attrib.type == ::cudaMemoryTypeUnregistered || __ptr_attrib.type == ::cudaMemoryTypeHost
            || __ptr_attrib.type == ::cudaMemoryTypeManaged;
#    else // ^^^ _CCCL_HIP_COMPILATION() ^^^ / vvv !_CCCL_HIP_COMPILATION() vvv
        ::CUmemorytype __type{};
        const auto __status =
          ::cuda::__driver::__pointerGetAttributeNoThrow<::CU_POINTER_ATTRIBUTE_MEMORY_TYPE>(__type, __p1);
        return (__status != ::cudaSuccess) || __type == ::CU_MEMORYTYPE_HOST;
#    endif // !_CCCL_HIP_COMPILATION()
      }
      return true;
    }
    else
#  endif // _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()
    {
      return true; // cannot be verified
    }
  }
#endif // !_CCCL_COMPILER(NVRTC)

public:
  using offset_policy    = __host_accessor<typename _Accessor::offset_policy>;
  using data_handle_type = __data_handle_type;
  using reference        = typename _Accessor::reference;
  using element_type     = typename _Accessor::element_type;

  _CCCL_TEMPLATE(class _Accessor2 = _Accessor)
  _CCCL_REQUIRES(::cuda::std::is_default_constructible_v<_Accessor2>)
  _CCCL_API constexpr __host_accessor() noexcept(::cuda::std::is_nothrow_default_constructible_v<_Accessor2>)
      : _Accessor{}
  {}

  _CCCL_API constexpr __host_accessor(_Accessor&& __acc) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Accessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {}

  _CCCL_API constexpr __host_accessor(const _Accessor& __acc) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Accessor>)
      : _Accessor{__acc}
  {}

  template <typename _OtherAccessor>
  __host_accessor(const __device_accessor<_OtherAccessor>&) = delete;

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    ::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr __host_accessor(const __host_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    !::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr explicit __host_accessor(const __host_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, _OtherAccessor> _CCCL_AND(
    ::cuda::std::is_convertible_v<_OtherAccessor, _Accessor>))
  _CCCL_API constexpr __host_accessor(__host_accessor<_OtherAccessor>&& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, _OtherAccessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    !::cuda::std::is_convertible_v<_OtherAccessor, _Accessor>))
  _CCCL_API constexpr explicit __host_accessor(__host_accessor<_OtherAccessor>&& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, _OtherAccessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    ::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr __host_accessor(const __managed_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    !::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr explicit __host_accessor(const __managed_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {}

  _CCCL_API constexpr reference access(data_handle_type __p, size_t __i) const noexcept(__is_access_noexcept)
  {
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (_CCCL_VERIFY(false, "cuda::__host_accessor cannot be used in DEVICE code");),
      (_CCCL_ASSERT(__is_host_accessible_pointer(__p), "cuda::__host_accessor data handle is not a HOST pointer");))
    return _Accessor::access(__p, __i);
  }

  [[nodiscard]] _CCCL_API constexpr data_handle_type offset(data_handle_type __p, size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    return _Accessor::offset(__p, __i);
  }

  [[nodiscard]] _CCCL_API constexpr bool
  __detectably_invalid([[maybe_unused]] data_handle_type __p, size_t) const noexcept
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return __is_host_accessible_pointer(__p);), (return false;))
  }
};

/***********************************************************************************************************************
 * Device Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
class __device_accessor : public _Accessor
{
  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::__host_accessor/cuda::__device_accessor/cuda::__managed_accessor cannot be nested");

  using __data_handle_type = typename _Accessor::data_handle_type;

  static constexpr bool __is_access_noexcept =
    noexcept(::cuda::std::declval<_Accessor>().access(::cuda::std::declval<__data_handle_type>(), 0));

  static constexpr bool __is_offset_noexcept =
    noexcept(::cuda::std::declval<_Accessor>().offset(::cuda::std::declval<__data_handle_type>(), 0));

  [[nodiscard]] _CCCL_API static constexpr bool
  __is_device_accessible_pointer_from_host([[maybe_unused]] __data_handle_type __p) noexcept
  {
#if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()
    if constexpr (::cuda::std::contiguous_iterator<__data_handle_type>)
    {
      auto __p1 = ::cuda::std::to_address(__p);
#  if _CCCL_HIP_COMPILATION()
      // NOTE(HIP/AMD): see __is_host_accessible_pointer above for rationale
      // -- HIP doesn't expose cuPointerGetAttribute through cuda::__driver,
      // so use the cudaPointerGetAttributes runtime API shim instead.
      ::cudaPointerAttributes __ptr_attrib{};
      _CCCL_ASSERT_CUDA_API(::cudaPointerGetAttributes, "cudaPointerGetAttributes failed", &__ptr_attrib, __p1);
      return __ptr_attrib.type == ::cudaMemoryTypeDevice || __ptr_attrib.type == ::cudaMemoryTypeManaged;
#  else // ^^^ _CCCL_HIP_COMPILATION() ^^^ / vvv !_CCCL_HIP_COMPILATION() vvv
      ::CUmemorytype __type{};
      const auto __status =
        ::cuda::__driver::__pointerGetAttributeNoThrow<::CU_POINTER_ATTRIBUTE_MEMORY_TYPE>(__type, __p1);
      return (__status != ::cudaSuccess) || __type == ::CU_MEMORYTYPE_DEVICE;
#  endif // !_CCCL_HIP_COMPILATION()
    }
    else
#endif // _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()
    {
      return true; // cannot be verified
    }
  }

#if _CCCL_DEVICE_COMPILATION()

  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static constexpr bool
  __is_device_accessible_pointer_from_device(__data_handle_type __p) noexcept
  {
    return ::cuda::device::is_address_from(__p, ::cuda::device::address_space::global)
        || ::cuda::device::is_address_from(__p, ::cuda::device::address_space::shared)
#if !_CCCL_HIP_COMPILATION()
        // NOTE(HIP/AMD): __isConstant has no AMDGCN equivalent (see
        // <amd/amd_utils.h>); calling is_address_from(..., constant)
        // on HIP-device traps. __constant__ globals on AMDGCN live in
        // the global address space anyway and are reported by the
        // is_address_from(..., global) query above, so we skip the
        // constant / grid_constant / cluster_shared checks here on
        // HIP. The first two queries (global / shared) cover all
        // supported AMDGCN address spaces.
        || ::cuda::device::is_address_from(__p, ::cuda::device::address_space::constant)
        || ::cuda::device::is_address_from(__p, ::cuda::device::address_space::grid_constant)
        || ::cuda::device::is_address_from(__p, ::cuda::device::address_space::cluster_shared)
#endif // !_CCCL_HIP_COMPILATION()
        || ::cuda::device::is_address_from(__p, ::cuda::device::address_space::local);
  }

#endif // _CCCL_DEVICE_COMPILATION()

  _CCCL_API static constexpr void __check_device_pointer([[maybe_unused]] __data_handle_type __p) noexcept
  {
    NV_IF_TARGET(NV_IS_HOST,
                 (_CCCL_ASSERT(__is_device_accessible_pointer_from_host(__p), "The pointer is not device accessible");))
  }

public:
  using offset_policy    = __device_accessor<typename _Accessor::offset_policy>;
  using data_handle_type = __data_handle_type;
  using reference        = typename _Accessor::reference;
  using element_type     = typename _Accessor::element_type;

  _CCCL_TEMPLATE(class _Accessor2 = _Accessor)
  _CCCL_REQUIRES(::cuda::std::is_default_constructible_v<_Accessor2>)
  _CCCL_API constexpr __device_accessor() noexcept(::cuda::std::is_nothrow_default_constructible_v<_Accessor2>)
      : _Accessor{}
  {}

  _CCCL_API constexpr __device_accessor(_Accessor&& __acc) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Accessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {}

  _CCCL_API constexpr __device_accessor(const _Accessor& __acc) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Accessor>)
      : _Accessor{__acc}
  {}

  template <typename _OtherAccessor>
  __device_accessor(const __host_accessor<_OtherAccessor>&) = delete;

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    ::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr __device_accessor(const __device_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    !::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr explicit __device_accessor(const __device_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, _OtherAccessor> _CCCL_AND(
    ::cuda::std::is_convertible_v<_OtherAccessor, _Accessor>))
  _CCCL_API constexpr __device_accessor(__device_accessor<_OtherAccessor>&& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, _OtherAccessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, _OtherAccessor> _CCCL_AND(
    !::cuda::std::is_convertible_v<_OtherAccessor, _Accessor>))
  _CCCL_API constexpr explicit __device_accessor(__device_accessor<_OtherAccessor>&& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, _OtherAccessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    ::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr __device_accessor(const __managed_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    !::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr explicit __device_accessor(const __managed_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {}

  _CCCL_API constexpr reference access(data_handle_type __p, size_t __i) const noexcept(__is_access_noexcept)
  {
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (_CCCL_ASSERT(__is_device_accessible_pointer_from_device(__p), "The pointer is not device accessible");),
      (_CCCL_VERIFY(false, "cuda::device_accessor cannot be used in HOST code");))
    return _Accessor::access(__p, __i);
  }

  [[nodiscard]] _CCCL_API constexpr data_handle_type offset(data_handle_type __p, size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    return _Accessor::offset(__p, __i);
  }

  [[nodiscard]] _CCCL_API constexpr bool __detectably_invalid(data_handle_type __p, size_t) const noexcept
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return __is_device_accessible_pointer_from_host(__p);), (return false;))
  }
};

/***********************************************************************************************************************
 * Managed Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
class __managed_accessor : public _Accessor
{
  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::__host_accessor/cuda::__device_accessor/cuda::__managed_accessor cannot be nested");

  using __data_handle_type = typename _Accessor::data_handle_type;

  static constexpr bool __is_access_noexcept =
    noexcept(::cuda::std::declval<_Accessor>().access(::cuda::std::declval<__data_handle_type>(), 0));

  static constexpr bool __is_offset_noexcept =
    noexcept(::cuda::std::declval<_Accessor>().offset(::cuda::std::declval<__data_handle_type>(), 0));

  [[nodiscard]] _CCCL_API static constexpr bool __is_managed_pointer([[maybe_unused]] __data_handle_type __p) noexcept
  {
#if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()
    if constexpr (::cuda::std::contiguous_iterator<__data_handle_type>)
    {
#if _CCCL_HIP_COMPILATION()
      // NOTE(HIP/AMD): the upstream "devicePointer == hostPointer" heuristic
      // works on HIP too — hipMallocManaged() populates both pointers to the
      // same address, plain hipMalloc() leaves hostPointer null, and
      // unregistered host memory leaves both null. The only HIP-specific gap
      // is that __managed__ global variables are currently never reported as
      // managed by HIP's hipPointerGetAttributes() (they show up with both
      // pointers null), so they can't be used with cuda::managed_mdspan on
      // HIP until that is fixed in the HIP runtime.
      ::cudaPointerAttributes __ptr_attrib{};
      auto __p1 = ::cuda::std::to_address(__p);
      _CCCL_ASSERT_CUDA_API(::cudaPointerGetAttributes, "cudaPointerGetAttributes failed", &__ptr_attrib, __p1);
      return __ptr_attrib.devicePointer != nullptr && __ptr_attrib.hostPointer == __ptr_attrib.devicePointer;
#else // ^^^ _CCCL_HIP_COMPILATION() ^^^ / vvv !_CCCL_HIP_COMPILATION() vvv
      const auto __p1 = ::cuda::std::to_address(__p);
      bool __is_managed{};
      const auto __status =
        ::cuda::__driver::__pointerGetAttributeNoThrow<::CU_POINTER_ATTRIBUTE_IS_MANAGED>(__is_managed, __p1);
      return (__status != ::cudaSuccess) || __is_managed;
#endif // !_CCCL_HIP_COMPILATION()
    }
    else
#endif // _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()
    {
      return true; // cannot be verified
    }
  }

  _CCCL_API static constexpr void __check_managed_pointer([[maybe_unused]] __data_handle_type __p) noexcept
  {
    _CCCL_ASSERT(__is_managed_pointer(__p), "cuda::__managed_accessor data handle is not a MANAGED pointer");
  }

public:
  using offset_policy    = __managed_accessor<typename _Accessor::offset_policy>;
  using data_handle_type = __data_handle_type;
  using reference        = typename _Accessor::reference;
  using element_type     = typename _Accessor::element_type;

  _CCCL_TEMPLATE(class _Accessor2 = _Accessor)
  _CCCL_REQUIRES(::cuda::std::is_default_constructible_v<_Accessor2>)
  _CCCL_API constexpr __managed_accessor() noexcept(::cuda::std::is_nothrow_default_constructible_v<_Accessor2>)
      : _Accessor{}
  {}

  _CCCL_API constexpr __managed_accessor(_Accessor&& __acc) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Accessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {}

  _CCCL_API constexpr __managed_accessor(const _Accessor& __acc) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Accessor>)
      : _Accessor{__acc}
  {}

  template <typename _OtherAccessor>
  __managed_accessor(const __host_accessor<_OtherAccessor>&) = delete;

  template <typename _OtherAccessor>
  __managed_accessor(const __device_accessor<_OtherAccessor>&) = delete;

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    ::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr __managed_accessor(const __managed_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    !::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr explicit __managed_accessor(const __managed_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, _OtherAccessor> _CCCL_AND(
    ::cuda::std::is_convertible_v<_OtherAccessor, _Accessor>))
  _CCCL_API constexpr __managed_accessor(__managed_accessor<_OtherAccessor>&& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, _OtherAccessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, _OtherAccessor> _CCCL_AND(
    !::cuda::std::is_convertible_v<_OtherAccessor, _Accessor>))
  _CCCL_API constexpr explicit __managed_accessor(__managed_accessor<_OtherAccessor>&& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, _OtherAccessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {}

  _CCCL_API constexpr reference access(data_handle_type __p, size_t __i) const noexcept(__is_access_noexcept)
  {
    NV_IF_TARGET(NV_IS_HOST, (__check_managed_pointer(__p);))
    return _Accessor::access(__p, __i);
  }

  [[nodiscard]] _CCCL_API constexpr data_handle_type offset(data_handle_type __p, size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    return _Accessor::offset(__p, __i);
  }

  [[nodiscard]] _CCCL_API constexpr bool
  __detectably_invalid([[maybe_unused]] data_handle_type __p, size_t) const noexcept
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return __is_managed_pointer(__p);), (return false;))
  }
};

/***********************************************************************************************************************
 * Accessibility Traits
 **********************************************************************************************************************/

template <typename>
inline constexpr bool is_host_accessible_v = false;

template <typename>
inline constexpr bool is_device_accessible_v = false;

template <typename _Accessor>
inline constexpr bool is_host_accessible_v<__host_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_host_accessible_v<__managed_accessor<_Accessor>> = true;

template <template <typename> class _TClass, typename _Accessor>
inline constexpr bool is_host_accessible_v<_TClass<_Accessor>> = is_host_accessible_v<_Accessor>;

template <typename _Accessor>
inline constexpr bool is_device_accessible_v<__device_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_device_accessible_v<__managed_accessor<_Accessor>> = true;

template <template <typename> class _TClass, typename _Accessor>
inline constexpr bool is_device_accessible_v<_TClass<_Accessor>> = is_device_accessible_v<_Accessor>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MDSPAN_HOST_DEVICE_ACCESSOR_H
