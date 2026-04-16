//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _CUDA___HIERARCHY_THREAD_LEVEL_H
#define _CUDA___HIERARCHY_THREAD_LEVEL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/__hierarchy/native_hierarchy_level_base.h>
// <<<<<<< OLD CODE from 6936be391d (a76de88057) - COMMENTED OUT
// #  include <cuda/std/__concepts/concept_macros.h>
// #  include <cuda/std/__mdspan/extents.h>
// #  include <cuda/std/__type_traits/is_integer.h>
//
// #  if _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()
// #    include <cuda/__ptx/instructions/get_sreg.h>
// #  endif // _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()
// =======
// >>>>>>> END NEW CODE (a76de88057)

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

struct _CCCL_DECLSPEC_EMPTY_BASES thread_level : __native_hierarchy_level_base<thread_level>
{
  using __product_type  = unsigned;
  using __allowed_above = __allowed_levels<block_level>;
  using __allowed_below = __allowed_levels<>;

  using __next_native_level = block_level;
// <<<<<<< OLD CODE from 6936be391d (a76de88057) - COMMENTED OUT
//
//   using __base_type = __native_hierarchy_level_base<thread_level>;
//
// // <<<<<<< OLD CODE from a65c61eebf (628a762319) - COMMENTED OUT
// // #  if _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()
// //   using __base_type::index_as;
// // =======
// #  if _CCCL_CUDA_COMPILATION()
// // >>>>>>> END NEW CODE (628a762319)
//   using __base_type::rank_as;
//
//   // interactions with warp level
//
//   // NOTE(HIP/AMD): the warp/wavefront size is 32 on NVIDIA but is wave32 or
//   // wave64 depending on the AMD GPU architecture, so use _CCCL_HIP_WAVE_SIZE.
//   _CCCL_TEMPLATE(class _Tp)
//   _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
// // <<<<<<< OLD CODE from a65c61eebf (628a762319) - COMMENTED OUT
// // // <<<<<<< OLD CODE from ca49929d59 (89b06d96af) - COMMENTED OUT
// // // #  if _CCCL_HIP_COMPILATION()
// // //   [[nodiscard]]
// // //   _CCCL_DEVICE_API static constexpr ::cuda::std::extents<_Tp, _CCCL_HIP_WAVE_SIZE> extents_as(const warp_level&) noexcept
// // // #  else // ^^^ _CCCL_HIP_COMPILATION() ^^^ / vvv !_CCCL_HIP_COMPILATION() vvv
// // //   [[nodiscard]]
// // //   _CCCL_DEVICE_API static constexpr ::cuda::std::extents<_Tp, 32> extents_as(const warp_level&) noexcept
// // // #  endif // !_CCCL_HIP_COMPILATION()
// // //   {
// // //     return {};
// // //   }
// // //
// // //   _CCCL_TEMPLATE(class _Tp)
// // //   _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
// // // =======
// // // >>>>>>> END NEW CODE (89b06d96af)
// //   [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> index_as(const warp_level&) noexcept
// //   {
// //     return {static_cast<_Tp>(::cuda::ptx::get_sreg_laneid()), 0, 0};
// //   }
// //
// //   _CCCL_TEMPLATE(class _Tp)
// //   _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
// // =======
// // >>>>>>> END NEW CODE (628a762319)
//   [[nodiscard]] _CCCL_DEVICE_API static _Tp rank_as(const warp_level&) noexcept
//   {
//     return static_cast<_Tp>(::cuda::ptx::get_sreg_laneid());
//   }
// #  endif // _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()
// =======
// >>>>>>> END NEW CODE (a76de88057)
};

_CCCL_GLOBAL_CONSTANT thread_level gpu_thread;

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#endif // _CUDA___HIERARCHY_THREAD_LEVEL_H
