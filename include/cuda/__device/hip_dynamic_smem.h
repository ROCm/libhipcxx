// MIT License
//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _CUDA___DEVICE_HIP_DYNAMIC_SMEM_H
#define _CUDA___DEVICE_HIP_DYNAMIC_SMEM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// NOTE(HIP/AMD): On the AMDGCN device pass, prefer the authoritative HSA runtime
// definition of the AQL kernel dispatch packet so the field offset always tracks
// the ABI. When the HSA headers are unavailable (e.g. HIPRTC or a minimal
// toolchain) fall back to a self-contained, ABI-pinned mirror further below.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(_CCCL_COMPILER_HIPRTC) && __has_include(<hsa/hsa.h>)
#  include <hsa/hsa.h>
#  define _CCCL_HAS_HSA_KERNEL_DISPATCH_PACKET
#endif

#include <cuda/std/__cstddef/types.h> // offsetof
#include <cuda/std/cstdint>           // uint16_t, uint32_t

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#if defined(__HIP_DEVICE_COMPILE__)

#  if !defined(_CCCL_HAS_HSA_KERNEL_DISPATCH_PACKET)
//! @brief Mirror of the HSA AQL kernel dispatch packet, truncated to the prefix
//! ending at @c group_segment_size. The layout is fixed by the HSA ABI; the
//! static_assert in @c __hip_group_segment_size turns any future divergence into
//! a compile error instead of a silent misread. Used only when the real
//! <hsa/hsa.h> definition is unavailable.
struct __hsa_kernel_dispatch_packet_prefix
{
  ::cuda::std::uint16_t __header;
  ::cuda::std::uint16_t __setup;
  ::cuda::std::uint16_t __workgroup_size_x;
  ::cuda::std::uint16_t __workgroup_size_y;
  ::cuda::std::uint16_t __workgroup_size_z;
  ::cuda::std::uint16_t __reserved0;
  ::cuda::std::uint32_t __grid_size_x;
  ::cuda::std::uint32_t __grid_size_y;
  ::cuda::std::uint32_t __grid_size_z;
  ::cuda::std::uint32_t __private_segment_size;
  ::cuda::std::uint32_t __group_segment_size;
};
#  endif // !_CCCL_HAS_HSA_KERNEL_DISPATCH_PACKET

//! @brief Returns the currently executing kernel's @c group_segment_size from the
//! HSA AQL kernel dispatch packet: the total (static + dynamic) LDS/shared-memory
//! bytes reserved for this launch -- AMDGCN's equivalent of NVPTX
//! @c %total_smem_size. AMDGCN has no dynamic-shared-memory special register, so
//! callers recover the dynamic portion by subtracting the static size
//! (@c __builtin_amdgcn_groupstaticsize()).
//!
//! When the HSA headers are available the field is read directly through the
//! authoritative @c hsa_kernel_dispatch_packet_t type; otherwise it is read via
//! @c offsetof on the ABI-pinned mirror, so the offset is never hard-coded.
[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __hip_group_segment_size() noexcept
{
#  if defined(_CCCL_HAS_HSA_KERNEL_DISPATCH_PACKET)
  // Authoritative HSA type available: read group_segment_size directly.
  return static_cast<const hsa_kernel_dispatch_packet_t*>(__builtin_amdgcn_dispatch_ptr())->group_segment_size;
#  else
  // Fallback: read via the ABI-pinned mirror of the packet prefix. offsetof gives
  // the field position within the mirror; the static_assert ties the mirror to the
  // fixed HSA ABI (group_segment_size @ 28) so a bad edit to the mirror fails to
  // compile instead of silently misreading the real packet.
  static_assert(offsetof(__hsa_kernel_dispatch_packet_prefix, __group_segment_size) == 28,
                "HSA AQL kernel dispatch packet layout changed unexpectedly");
  const char* __packet                       = static_cast<const char*>(__builtin_amdgcn_dispatch_ptr());
  ::cuda::std::uint32_t __group_segment_size  = 0;
  __builtin_memcpy(&__group_segment_size,
                   __packet + offsetof(__hsa_kernel_dispatch_packet_prefix, __group_segment_size),
                   sizeof(__group_segment_size));
  return __group_segment_size;
#  endif // !_CCCL_HAS_HSA_KERNEL_DISPATCH_PACKET
}

#endif // __HIP_DEVICE_COMPILE__

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___DEVICE_HIP_DYNAMIC_SMEM_H
