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

#ifndef _CUDA___HIERARCHY_GET_LAUNCH_DIMENSIONS_H
#define _CUDA___HIERARCHY_GET_LAUNCH_DIMENSIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#  include <cuda/__hierarchy/block_level.h>
#  include <cuda/__hierarchy/cluster_level.h>
#  include <cuda/__hierarchy/grid_level.h>
#  include <cuda/__hierarchy/thread_level.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/std/tuple>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

/**
 * @brief Returns a tuple of dim3 compatible objects that can be used to launch
 * a kernel
 *
 * This function returns a tuple of hierarchy_query_result objects that contain
 * dimensions from the supplied hierarchy, that can be used to launch that
 * hierarchy. It is meant to allow for easy usage of hierarchy dimensions with
 * the <<<>>> launch syntax or cudaLaunchKernelEx in case of a cluster launch.
 * Contained hierarchy_query_result objects are results of extents() member
 * function on the hierarchy passed in. The returned tuple has three elements if
 * cluster_level is present in the hierarchy (extents(block, grid),
 * extents(cluster, block), extents(thread, block)). Otherwise it contains only
 * two elements, without the middle one related to the cluster.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * using namespace cuda;
 *
 * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(),
 * block_dims<8, 8, 8>()); auto [grid_dimensions, cluster_dimensions,
 * block_dimensions] = get_launch_dimensions(hierarchy);
 * assert(grid_dimensions.x == 256);
 * assert(cluster_dimensions.x == 4);
 * assert(block_dimensions.x == 8);
 * assert(block_dimensions.y == 8);
 * assert(block_dimensions.z == 8);
 * @endcode
 * @par
 *
 * @param __hierarchy
 *  Hierarchy that the launch dimensions are requested for
 */
template <class _BottomLevel, class... _LevelDescs>
[[nodiscard]] _CCCL_HOST_API constexpr auto
get_launch_dimensions(const hierarchy<_BottomLevel, _LevelDescs...>& __hierarchy)
{
  if constexpr (hierarchy<_BottomLevel, _LevelDescs...>::has_level(cluster))
  {
    return ::cuda::std::make_tuple(
      ::dim3{block.dims(grid, __hierarchy)},
      ::dim3{block.dims(cluster, __hierarchy)},
      ::dim3{gpu_thread.dims(block, __hierarchy)});
  }
  else
  {
    return ::cuda::std::make_tuple(::dim3{block.dims(grid, __hierarchy)}, ::dim3{gpu_thread.dims(block, __hierarchy)});
  }
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#endif // _CUDA___HIERARCHY_GET_LAUNCH_DIMENSIONS_H
