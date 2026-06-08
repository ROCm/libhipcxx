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

#ifndef _CUDA_STD___PSTL_CUDA_HIPCUB_H
#define _CUDA_STD___PSTL_CUDA_HIPCUB_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// NOTE(HIP/AMD): the libcudacxx PSTL "cuda" backend (cuda/std/__pstl/cuda/*.h) is
// written against NVIDIA CUB -- ::cub::Device* and the CUB_NS_QUALIFIER::detail::
// transform internals. On HIP we reuse that backend code without forking it by
// providing a curated ::cub namespace that re-exports hipCUB's device algorithms
// and supplies the few pieces hipCUB does not expose:
//   * cub::DeviceTransform::Generate  (hipCUB's DeviceTransform has Transform only)
//   * cub::detail::transform::{requires_stable_address, always_true_predicate,
//     dispatch}  (CUB-internal; not part of hipCUB's public API)
// The cuda* runtime symbols (cudaStream_t, cudaError_t, cudaStreamPerThread,
// cudaMemcpy*) are already hip-aliased globally via <cuda/std/detail/__config> ->
// <libhipcxx/__amd/cuda_runtime.h>, and _CCCL_TRY_CUDA_API works on HIP.
//
// dispatch()/Generate() are implemented directly over rocprim::transform with a
// counting+discard iterator and an explicit index op (the same reliable pattern
// hipCUB uses internally), NOT over hipcub::DeviceTransform::Transform's tuple
// overload (whose op-argument convention differs from CUB's unpacked one).
#if _CCCL_HIP_COMPILATION()

#  include <hipcub/hipcub.hpp>

#  include <rocprim/device/device_transform.hpp>
#  include <rocprim/iterator/counting_iterator.hpp>
#  include <rocprim/iterator/discard_iterator.hpp>
#  include <rocprim/types.hpp>

#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__tuple_dir/apply.h>
#  include <cuda/std/__type_traits/is_same.h>

// CUB_NS_QUALIFIER is a CUB-only macro (cub/util_namespace.cuh). Point it at our
// curated ::cub namespace so CUB_NS_QUALIFIER::detail::transform::... resolves.
#  ifndef CUB_NS_QUALIFIER
#    define CUB_NS_QUALIFIER ::cub
#  endif // CUB_NS_QUALIFIER

namespace cub
{
// Re-export the hipCUB device algorithms the PSTL backend / benches use as-is.
using ::hipcub::ArgMax;
using ::hipcub::ArgMin;
using ::hipcub::DeviceCopy;
using ::hipcub::DeviceFor;
using ::hipcub::DeviceReduce;
using ::hipcub::DeviceRunLengthEncode;

// hipCUB's DeviceTransform exposes Transform but not Generate; add Generate while
// inheriting the rest.
struct DeviceTransform : ::hipcub::DeviceTransform
{
  // generate_n.h passes a cuda::stream_ref (matching CUB's Generate(out, n, op, stream));
  // extract the native HIP stream for rocprim.
  template <class _OutIt, class _OffsetT, class _GenOp>
  static hipError_t Generate(_OutIt __out, _OffsetT __count, _GenOp __gen, ::cuda::stream_ref __stream)
  {
    return ::rocprim::transform(
      ::rocprim::counting_iterator<_OffsetT>(0),
      ::rocprim::discard_iterator(),
      __count,
      [__out, __gen] __host__ __device__(_OffsetT __i) mutable {
        __out[__i] = __gen();
        return ::rocprim::empty_type{};
      },
      __stream.get());
  }
};

namespace detail
{
namespace transform
{
enum class requires_stable_address
{
  no,
  yes
};

struct always_true_predicate
{
  template <class... _As>
  __host__ __device__ constexpr bool operator()(_As&&...) const noexcept
  {
    return true;
  }
};

template <class _First, class... _Rest>
__host__ __device__ _First&& __first_of(_First&& __f, _Rest&&...) noexcept
{
  return static_cast<_First&&>(__f);
}

// out[i] = pred(in0[i], in1[i], ...) ? op(in0[i], in1[i], ...) : in0[i]
// (CUB transform semantics; always_true_predicate -> plain transform).
template <requires_stable_address /*unused: rocprim handles addressing*/,
          class _InTuple,
          class _OutIt,
          class _OffsetT,
          class _Pred,
          class _Op>
hipError_t dispatch(_InTuple __inputs, _OutIt __out, _OffsetT __count, _Pred __pred, _Op __op, hipStream_t __stream)
{
  return ::cuda::std::apply(
    [&](auto... __its) {
      return ::rocprim::transform(
        ::rocprim::counting_iterator<_OffsetT>(0),
        ::rocprim::discard_iterator(),
        __count,
        [__out, __pred, __op, __its...] __host__ __device__(_OffsetT __i) mutable {
          // Materialize each input to its VALUE type before calling op/pred: thrust
          // device iterators yield proxy references (device_reference<T>), but CUB
          // passes values to the op/predicate -- passing the proxy would deduce the
          // op/pred template parameter as device_reference<T> and break code like
          // static_cast<T>(x).
          auto __get = [__i](auto __it) {
            return static_cast<::cuda::std::iter_value_t<decltype(__it)>>(__it[__i]);
          };
          // For plain transform (always_true_predicate) the pass-through branch is
          // dead; omit it so op's result type need not be compatible with the input
          // (e.g. transform<In->Out>). For a real predicate use if/else (not ?:) so
          // the two branches need no common type.
          if constexpr (::cuda::std::is_same_v<_Pred, always_true_predicate>)
          {
            __out[__i] = __op(__get(__its)...);
          }
          else if (__pred(__get(__its)...))
          {
            __out[__i] = __op(__get(__its)...);
          }
          else
          {
            __out[__i] = __first_of(__get(__its)...);
          }
          return ::rocprim::empty_type{};
        },
        __stream);
    },
    __inputs);
}
} // namespace transform
} // namespace detail
} // namespace cub

#endif // _CCCL_HIP_COMPILATION()

#endif // _CUDA_STD___PSTL_CUDA_HIPCUB_H
