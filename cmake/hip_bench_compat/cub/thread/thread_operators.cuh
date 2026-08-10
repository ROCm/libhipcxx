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

// HIP compat shim: forward this NVIDIA CUB header to hipCUB so the unmodified
// upstream benchmark / nvbench_helper sources compile on HIP (see README.md).
#pragma once
#include <hipcub/hipcub.hpp>
// Real CUB makes <cuda/functional> (cuda::proclaims_copyable_arguments, ...) visible
// transitively; hipCUB does not, and nvbench_helper.cuh relies on it.
#include <cuda/functional>
// Provides the curated ::cub namespace (re-exports hipCUB + adds the pieces hipCUB
// lacks: cub::DeviceTransform::Generate and cub::detail::transform::*).
#include <cuda/std/__pstl/cuda/__hipcub.h>

// TODO(HIP/AMD): remove this alias block once hipCUB is rebased on CUB 3.x.
//
// CUB 3.x renamed the arg-extremum reduction functors to lowercase
// `cub::detail::arg_min` / `cub::detail::arg_max`. nvbench_helper.cuh (3.4.0)
// references them directly (NVBENCH_DECLARE_TYPE_STRINGS + the
// min_element/max_element benches). hipCUB still ships the old capitalized
// `hipcub::ArgMin` / `hipcub::ArgMax` (plain structs with a templated
// operator()(KeyValuePair,KeyValuePair)), so map the new names onto them.
//
// Removal condition: hipCUB exports `hipcub::detail::arg_min` / `arg_max` (i.e.
// it has picked up the CUB 3.x rename). At that point the two `using`
// declarations below become redundant with the re-exported ::cub namespace from
// <cuda/std/__pstl/cuda/__hipcub.h> and should be deleted, not re-pointed.
namespace cub
{
namespace detail
{
using arg_min = ::hipcub::ArgMin;
using arg_max = ::hipcub::ArgMax;
} // namespace detail
} // namespace cub
