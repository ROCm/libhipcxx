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
#if _CCCL_HIP_COMPILATION() && !defined(_CCCL_COMPILER_HIPRTC)

#  include <hipcub/hipcub.hpp>
#  include <hipcub/device/device_adjacent_difference.hpp>
#  include <hipcub/device/device_merge.hpp>
#  include <hipcub/device/device_merge_sort.hpp>
#  include <hipcub/device/device_partition.hpp>
#  include <hipcub/device/device_radix_sort.hpp>
#  include <hipcub/device/device_scan.hpp>
#  include <hipcub/device/device_select.hpp>

#  include <rocprim/device/device_partition.hpp>
#  include <rocprim/device/device_transform.hpp>
#  include <rocprim/iterator/counting_iterator.hpp>
#  include <rocprim/iterator/discard_iterator.hpp>
#  include <rocprim/iterator/transform_output_iterator.hpp>
#  include <rocprim/types.hpp>

#  include <cuda/__functional/always_true_false.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__tuple_dir/apply.h>
#  include <cuda/std/__type_traits/is_arithmetic.h>
#  include <cuda/std/__type_traits/is_one_of.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/remove_cvref.h>

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
// hipcub's DeviceFor::ForEachN takes a hipStream_t, but for_each_n.h (3.4.0)
// passes the execution policy as an environment. Wrap DeviceFor to accept the
// env: extract the native HIP stream and forward to hipcub. The using-declaration
// keeps hipcub's native stream / temp-storage overloads available.
struct DeviceFor : ::hipcub::DeviceFor
{
  using ::hipcub::DeviceFor::ForEachN;

  template <class _RandomAccessIt, class _OffsetT, class _Op, class _Env>
  static hipError_t ForEachN(_RandomAccessIt __first, _OffsetT __n, _Op __op, _Env __env)
  {
    auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{hipStream_t{}}, __env);
    return ::hipcub::DeviceFor::ForEachN(__first, __n, ::cuda::std::move(__op), __stream.get());
  }
};
using ::hipcub::DeviceRunLengthEncode;
// DeviceSelect: hipcub's If/Flagged signatures match CUB's exactly (no template
// NumItemsT ambiguity — hipcub uses int64_t), so a plain using is sufficient.
using ::hipcub::DeviceSelect;

// DeviceReduce: inherit hipCUB's implementation and add predicate-dropping wrappers
// for ArgMax and ArgMin.  The PSTL max_element / min_element backends call the
// new CUB two-output-iterator API with an extra comparator argument:
//   ArgMax(d_temp, bytes, d_in, d_value_out, d_index_out, count, pred, stream)
// where d_value_out = cuda::discard_iterator (ignore the extremum value, keep only
// the index) and d_index_out = size_t* (where the index is written).
//
// hipCUB's new 7-arg ArgMax creates a zip_iterator<(idx_out, val_out)> internally,
// which fails when val_out is cuda::discard_iterator (deref type is void).
// We work around this by using hipCUB's legacy 6-arg ArgMax which writes a
// KeyValuePair<OffsetT, T> to a single output, and extract the key (index) into
// d_idx_out via rocprim::transform_output_iterator.
//
// hipCUB's ArgMax/ArgMin do not accept a user comparator (the arg-max/min functor
// is fixed internally, comparing with the built-in operator< / operator>).  Only
// the DEFAULT cuda::std::less<> comparator that max_element / min_element pass has
// semantics identical to hipCUB's fixed ArgMax / ArgMin.  A non-default comparator
// (e.g. cuda::std::greater<>, which flips max_element into a minimum search) would
// be silently ignored and produce a WRONG result, so we reject it at compile time
// with a static_assert rather than miscompile (see __hipcub_argextremum_pred_ok).
//
// TransformReduce is forwarded unchanged — hipCUB's signature matches CUB's, and
// its reduction/transform ops ARE forwarded (not dropped), so any op is supported.
// Both cub::DeviceTransform::TransformIf overloads below drop the predicate and
// forward to a plain hipcub Transform, which is only correct when the predicate
// selects everything. Every call site passes ::cuda::always_true today, but
// copy_n's _UnaryPred is a defaulted template parameter, so a frontend that ever
// exposes it would silently get an unconditional transform. Reject anything else
// at compile time instead, exactly as __hipcub_argextremum_pred_ok does above.
template <class _Pred>
inline constexpr bool __hipcub_transformif_pred_ok =
  ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Pred>, ::cuda::always_true>;

template <class _Pred, class _T>
inline constexpr bool __hipcub_argextremum_pred_ok =
  ::cuda::std::__is_one_of_v<::cuda::std::remove_cvref_t<_Pred>,
                             ::cuda::std::less<>,
                             ::cuda::std::less<_T>>;

struct DeviceReduce : ::hipcub::DeviceReduce
{
  // ArgMax with predicate (8-arg CUB new-style).
  // Ignores d_value_out (discarded) and pred; extracts only the index into d_idx_out.
  template <class _InIt, class _ValOutIt, class _IdxOutIt, class _OffsetT, class _Pred>
  static hipError_t ArgMax(
    void* __d_temp, size_t& __bytes, _InIt __in, _ValOutIt /*__val_out*/, _IdxOutIt __idx_out,
    _OffsetT __n, _Pred /*__pred*/, hipStream_t __stream)
  {
    using _T     = typename ::cuda::std::iterator_traits<_InIt>::value_type;
    static_assert(__hipcub_argextremum_pred_ok<_Pred, _T>,
                  "cuda::std::max_element on HIP (hipCUB ArgMax shim) only supports the default "
                  "cuda::std::less comparator; hipCUB's ArgMax has a fixed extremum functor and "
                  "cannot honor a custom comparator. A non-default comparator would be silently "
                  "ignored, so it is rejected here instead of miscompiling.");
    using _KVOut = ::hipcub::KeyValuePair<_OffsetT, _T>;
    // Build a transform_output_iterator that extracts only the key (index) into d_idx_out.
    auto __kv_to_idx = [__idx_out] __host__ __device__(const _KVOut& __kv) {
      *__idx_out = static_cast<typename ::cuda::std::iterator_traits<_IdxOutIt>::value_type>(__kv.key);
      return ::rocprim::empty_type{};
    };
    auto __out = ::rocprim::make_transform_output_iterator(
      ::rocprim::discard_iterator(), __kv_to_idx);
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    return ::hipcub::DeviceReduce::ArgMax(__d_temp, __bytes, __in, __out, __n, __stream);
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
  }

  // ArgMin with predicate (8-arg CUB new-style).
  // Ignores d_value_out (discarded) and pred; extracts only the index into d_idx_out.
  template <class _InIt, class _ValOutIt, class _IdxOutIt, class _OffsetT, class _Pred>
  static hipError_t ArgMin(
    void* __d_temp, size_t& __bytes, _InIt __in, _ValOutIt /*__val_out*/, _IdxOutIt __idx_out,
    _OffsetT __n, _Pred /*__pred*/, hipStream_t __stream)
  {
    using _T     = typename ::cuda::std::iterator_traits<_InIt>::value_type;
    static_assert(__hipcub_argextremum_pred_ok<_Pred, _T>,
                  "cuda::std::min_element on HIP (hipCUB ArgMin shim) only supports the default "
                  "cuda::std::less comparator; hipCUB's ArgMin has a fixed extremum functor and "
                  "cannot honor a custom comparator. A non-default comparator would be silently "
                  "ignored, so it is rejected here instead of miscompiling.");
    using _KVOut = ::hipcub::KeyValuePair<_OffsetT, _T>;
    auto __kv_to_idx = [__idx_out] __host__ __device__(const _KVOut& __kv) {
      *__idx_out = static_cast<typename ::cuda::std::iterator_traits<_IdxOutIt>::value_type>(__kv.key);
      return ::rocprim::empty_type{};
    };
    auto __out = ::rocprim::make_transform_output_iterator(
      ::rocprim::discard_iterator(), __kv_to_idx);
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    return ::hipcub::DeviceReduce::ArgMin(__d_temp, __bytes, __in, __out, __n, __stream);
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
  }
};

// Re-export sort types/primitives from hipCUB.
using ::hipcub::DoubleBuffer;

// DeviceRadixSort: thin wrappers with exact pointer-compatible signatures.
// sort.h takes function pointers of type:
//   hipError_t (*)(void*, size_t&, cub::DoubleBuffer<_Tp>&, size_t, int, int, hipStream_t)
// hipcub::DeviceRadixSort::SortKeys has a template NumItemsT, so taking its address is
// ambiguous. Provide thin non-template forwarding wrappers.
struct DeviceRadixSort
{
  template <class _Tp>
  static hipError_t SortKeys(
    void* __d_temp, size_t& __temp_bytes, ::hipcub::DoubleBuffer<_Tp>& __keys,
    size_t __n, int __begin_bit, int __end_bit, hipStream_t __stream)
  {
    return ::hipcub::DeviceRadixSort::SortKeys(
      __d_temp, __temp_bytes, __keys, __n, __begin_bit, __end_bit, __stream);
  }

  template <class _Tp>
  static hipError_t SortKeysDescending(
    void* __d_temp, size_t& __temp_bytes, ::hipcub::DoubleBuffer<_Tp>& __keys,
    size_t __n, int __begin_bit, int __end_bit, hipStream_t __stream)
  {
    return ::hipcub::DeviceRadixSort::SortKeysDescending(
      __d_temp, __temp_bytes, __keys, __n, __begin_bit, __end_bit, __stream);
  }
};

// DeviceMergeSort: wraps hipcub's 2-phase (temp-storage) API into the new-style
// 4-arg CUB API that sort.h calls: SortKeys(iterator, count, comp, policy).
// The policy provides the stream (or falls back to hipStreamPerThread).
struct DeviceMergeSort
{
  template <class _KeyIterator, class _OffsetT, class _CompareOp, class _Policy>
  static hipError_t SortKeys(_KeyIterator __keys, _OffsetT __count, _CompareOp __comp,
                             const _Policy& __policy)
  {
    // Extract stream from policy (or default to hipStreamPerThread).
    ::cuda::stream_ref __sref =
      ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{hipStreamPerThread}, __policy);
    hipStream_t __stream = __sref.get();

    // Phase 1: query temp storage size.
    size_t __temp_bytes = 0;
    hipError_t __err    = ::hipcub::DeviceMergeSort::SortKeys(
      static_cast<void*>(nullptr), __temp_bytes, __keys, __count, __comp, __stream);
    if (__err != hipSuccess)
    {
      return __err;
    }

    // Phase 2: allocate temp storage and sort.
    void* __d_temp = nullptr;
    if (__temp_bytes > 0)
    {
      __err = ::hipMalloc(&__d_temp, __temp_bytes);
      if (__err != hipSuccess)
      {
        return __err;
      }
    }
    __err = ::hipcub::DeviceMergeSort::SortKeys(__d_temp, __temp_bytes, __keys, __count, __comp, __stream);
    if (__d_temp)
    {
      (void) ::hipFree(__d_temp);
    }
    return __err;
  }
};

// DeviceScan: wraps hipcub's 2-phase (temp-storage) API into the new-style CUB API
// that exclusive_scan.h and inclusive_scan.h call:
//   ExclusiveScan(d_in, d_out, op, init, count, policy)
//   InclusiveScan(d_in, d_out, op, count, policy)
//   InclusiveScanInit(d_in, d_out, op, init, count, policy)
// The policy provides the stream (or falls back to hipStreamPerThread).
struct DeviceScan
{
  template <class _InIt, class _OutIt, class _ScanOpT, class _InitT, class _OffsetT, class _Policy>
  static hipError_t ExclusiveScan(
    _InIt __in, _OutIt __out, _ScanOpT __op, _InitT __init, _OffsetT __count, const _Policy& __policy)
  {
    ::cuda::stream_ref __sref =
      ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{hipStreamPerThread}, __policy);
    hipStream_t __stream = __sref.get();

    size_t __temp_bytes = 0;
    hipError_t __err    = ::hipcub::DeviceScan::ExclusiveScan(
      static_cast<void*>(nullptr), __temp_bytes, __in, __out, __op, __init, __count, __stream);
    if (__err != hipSuccess)
    {
      return __err;
    }

    void* __d_temp = nullptr;
    if (__temp_bytes > 0)
    {
      __err = ::hipMalloc(&__d_temp, __temp_bytes);
      if (__err != hipSuccess)
      {
        return __err;
      }
    }
    __err = ::hipcub::DeviceScan::ExclusiveScan(__d_temp, __temp_bytes, __in, __out, __op, __init, __count, __stream);
    if (__d_temp)
    {
      (void) ::hipFree(__d_temp);
    }
    return __err;
  }

  template <class _InIt, class _OutIt, class _ScanOpT, class _OffsetT, class _Policy>
  static hipError_t InclusiveScan(_InIt __in, _OutIt __out, _ScanOpT __op, _OffsetT __count, const _Policy& __policy)
  {
    ::cuda::stream_ref __sref =
      ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{hipStreamPerThread}, __policy);
    hipStream_t __stream = __sref.get();

    size_t __temp_bytes = 0;
    hipError_t __err    = ::hipcub::DeviceScan::InclusiveScan(
      static_cast<void*>(nullptr), __temp_bytes, __in, __out, __op, __count, __stream);
    if (__err != hipSuccess)
    {
      return __err;
    }

    void* __d_temp = nullptr;
    if (__temp_bytes > 0)
    {
      __err = ::hipMalloc(&__d_temp, __temp_bytes);
      if (__err != hipSuccess)
      {
        return __err;
      }
    }
    __err = ::hipcub::DeviceScan::InclusiveScan(__d_temp, __temp_bytes, __in, __out, __op, __count, __stream);
    if (__d_temp)
    {
      (void) ::hipFree(__d_temp);
    }
    return __err;
  }

  template <class _InIt, class _OutIt, class _ScanOpT, class _InitT, class _OffsetT, class _Policy>
  static hipError_t InclusiveScanInit(
    _InIt __in, _OutIt __out, _ScanOpT __op, _InitT __init, _OffsetT __count, const _Policy& __policy)
  {
    ::cuda::stream_ref __sref =
      ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{hipStreamPerThread}, __policy);
    hipStream_t __stream = __sref.get();

    size_t __temp_bytes = 0;
    hipError_t __err    = ::hipcub::DeviceScan::InclusiveScanInit(
      static_cast<void*>(nullptr), __temp_bytes, __in, __out, __op, __init, __count, __stream);
    if (__err != hipSuccess)
    {
      return __err;
    }

    void* __d_temp = nullptr;
    if (__temp_bytes > 0)
    {
      __err = ::hipMalloc(&__d_temp, __temp_bytes);
      if (__err != hipSuccess)
      {
        return __err;
      }
    }
    __err =
      ::hipcub::DeviceScan::InclusiveScanInit(__d_temp, __temp_bytes, __in, __out, __op, __init, __count, __stream);
    if (__d_temp)
    {
      (void) ::hipFree(__d_temp);
    }
    return __err;
  }
};

// __can_use_radix_sort: mirrors CUB's variable template but without __half/__nv_bfloat16
// special-casing (HIP arithmetic types are exactly the standard arithmetic types).
template <class _InputIterator,
          class _BinaryPredicate,
          class _ValueType = ::cuda::std::iter_value_t<_InputIterator>>
inline constexpr bool __can_use_radix_sort =
  ::cuda::std::is_arithmetic_v<_ValueType>
  && ::cuda::std::__is_one_of_v<::cuda::std::remove_cvref_t<_BinaryPredicate>,
                                 ::cuda::std::less<>,
                                 ::cuda::std::less<_ValueType>,
                                 ::cuda::std::greater<>,
                                 ::cuda::std::greater<_ValueType>>;

// hipCUB's DeviceTransform exposes Transform but not Generate or TransformIf.
// Add both while inheriting the rest.
// NOTE(HIP/AMD): shift_right.h calls DeviceTransform::Transform with a
// cuda::std::tuple argument, but hipcub::DeviceTransform::Transform expects a
// hipcub::tuple (= rocprim::tuple). Since the inherited overloads won't match
// cuda::std::tuple, we add forwarding overloads that accept cuda::std::tuple and
// dispatch to the single-iterator (unwrapped) hipcub overloads.
struct DeviceTransform : ::hipcub::DeviceTransform
{
  // 5-arg form: Transform(cuda::std::tuple<Iters...>{...}, out, n, op, stream)
  // Used by shift_right.h non-overlapping paths.
  // shift_right.h always passes a 1-element tuple (single source iterator).
  // Unpack it and call the scalar (non-tuple) hipcub::DeviceTransform::Transform
  // overload to avoid the rocprim unpack_nary_op const-qualifier drop.
  template <class... _InIters, class _OutIt, class _OffsetT, class _Op>
  static hipError_t Transform(
    ::cuda::std::tuple<_InIters...> __inputs, _OutIt __out, _OffsetT __n, _Op __op, hipStream_t __stream)
  {
    static_assert(sizeof...(_InIters) == 1,
                  "cub::DeviceTransform::Transform(cuda::std::tuple): HIP shim supports only 1-element tuples "
                  "(shift_right use case); multi-input not yet implemented");
    return ::hipcub::DeviceTransform::Transform(
      ::cuda::std::get<0>(::cuda::std::move(__inputs)), __out, __n, __op, __stream);
  }

  // 7-arg form: Transform(d_temp, bytes, cuda::std::tuple<Iters...>{...}, out, n, op, stream)
  // Used by shift_right.h temporary-storage paths.
  template <class... _InIters, class _OutIt, class _OffsetT, class _Op>
  static hipError_t Transform(
    void* __d_temp, size_t& __bytes,
    ::cuda::std::tuple<_InIters...> __inputs, _OutIt __out, _OffsetT __n, _Op __op, hipStream_t __stream)
  {
    static_assert(sizeof...(_InIters) == 1,
                  "cub::DeviceTransform::Transform(d_temp, bytes, cuda::std::tuple): HIP shim supports only "
                  "1-element tuples (shift_right use case); multi-input not yet implemented");
    return ::hipcub::DeviceTransform::Transform(
      __d_temp, __bytes,
      ::cuda::std::get<0>(::cuda::std::move(__inputs)), __out, __n, __op, __stream);
  }
  // generate_n.h (3.4.0) passes the execution policy as an environment (CUB's
  // DeviceTransform::Generate(out, count, op, env)); extract the native HIP
  // stream from it for rocprim (falling back to the default stream). Also
  // accepts a plain cuda::stream_ref env (get_stream on a stream_ref is identity).
  template <class _OutIt, class _OffsetT, class _GenOp, class _Env>
  static hipError_t Generate(_OutIt __out, _OffsetT __count, _GenOp __gen, _Env __env)
  {
    auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{hipStream_t{}}, __env);
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

  // TransformIf: used in sort.h's radix copy-back with ::cuda::always_true{} predicate
  // (unconditional copy). Forward to hipcub::DeviceTransform::Transform dropping the
  // always-true predicate. The call site passes a single-input tuple, so use the tuple
  // overload of hipcub's Transform.
  // Signature expected by sort.h's _CCCL_TRY_CUDA_API call:
  //   TransformIf(tuple{src_ptr}, dst, count, ::cuda::always_true{}, identity{}, stream)
  // We forward (inputs, output, num_items, op, stream) dropping the predicate.
  template <class _InTuple, class _OutIt, class _OffsetT, class _Pred, class _Op>
  static hipError_t TransformIf(
    _InTuple __inputs, _OutIt __out, _OffsetT __count, _Pred /*__pred*/, _Op __op, hipStream_t __stream)
  {
    static_assert(__hipcub_transformif_pred_ok<_Pred>,
                  "cub::DeviceTransform::TransformIf: the HIP shim forwards to hipCUB's "
                  "unconditional Transform and cannot honour a predicate. Only "
                  "::cuda::always_true is supported.");
    // Convert cuda::std::tuple to hipcub::tuple by extracting the single element.
    // The call site passes cuda::std::tuple{src_ptr} (one input pointer).
    // hipcub::DeviceTransform::Transform(hipcub::tuple<Ins...>, out, n, op, stream)
    // Since we only ever get a single-input tuple from sort.h, use std::get<0> to unwrap
    // and call the single-iterator Transform overload instead (avoids tuple mismatch).
    return ::hipcub::DeviceTransform::Transform(
      ::cuda::std::get<0>(__inputs), __out, __count, __op, __stream);
  }

  // TransformIf with policy: used in copy_n.h's implementation with a predicate
  // defaulting to ::cuda::always_true{} (unconditional copy). The PSTL copy_n
  // frontend always calls without an explicit predicate so _Pred is always_true.
  // Extract the stream from the policy and forward to hipcub::DeviceTransform::Transform,
  // dropping the always-true predicate. Single-input tuple assumed (copy_n use case).
  // Signature expected by copy_n.h's _CCCL_TRY_CUDA_API call:
  //   TransformIf(tuple<In>{first}, result, count, pred, identity{}, policy)
  template <class _InTuple, class _OutIt, class _OffsetT, class _Pred, class _Op, class _Policy,
            // Exclude hipStream_t to avoid conflict with the stream overload above.
            ::cuda::std::enable_if_t<!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Policy>, hipStream_t>, int> = 0>
  static hipError_t TransformIf(
    _InTuple __inputs, _OutIt __out, _OffsetT __count, _Pred /*__pred*/, _Op __op, const _Policy& __policy)
  {
    static_assert(__hipcub_transformif_pred_ok<_Pred>,
                  "cub::DeviceTransform::TransformIf: the HIP shim forwards to hipCUB's "
                  "unconditional Transform and cannot honour a predicate. Only "
                  "::cuda::always_true is supported.");
    ::cuda::stream_ref __sref =
      ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{hipStreamPerThread}, __policy);
    return ::hipcub::DeviceTransform::Transform(
      ::cuda::std::get<0>(__inputs), __out, __count, __op, __sref.get());
  }
};

// DeviceAdjacentDifference: wraps hipCUB's 2-phase (temp-storage) API into the
// new-style 5-arg CUB API that adjacent_difference.h calls:
//   SubtractLeftCopy(in, out, count, op, policy)
// The policy provides the stream (or falls back to hipStreamPerThread).
struct DeviceAdjacentDifference
{
  template <class _InIt, class _OutIt, class _OffsetT, class _BinaryOp, class _Policy>
  static hipError_t SubtractLeftCopy(_InIt __in, _OutIt __out, _OffsetT __count, _BinaryOp __op,
                                     const _Policy& __policy)
  {
    ::cuda::stream_ref __sref =
      ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{hipStreamPerThread}, __policy);
    hipStream_t __stream = __sref.get();

    size_t __temp_bytes = 0;
    hipError_t __err    = ::hipcub::DeviceAdjacentDifference::SubtractLeftCopy(
      static_cast<void*>(nullptr), __temp_bytes, __in, __out, __count, __op, __stream);
    if (__err != hipSuccess)
    {
      return __err;
    }

    void* __d_temp = nullptr;
    if (__temp_bytes > 0)
    {
      __err = ::hipMalloc(&__d_temp, __temp_bytes);
      if (__err != hipSuccess)
      {
        return __err;
      }
    }
    __err = ::hipcub::DeviceAdjacentDifference::SubtractLeftCopy(
      __d_temp, __temp_bytes, __in, __out, __count, __op, __stream);
    if (__d_temp)
    {
      (void) ::hipFree(__d_temp);
    }
    return __err;
  }
};

// DeviceMerge: wraps hipCUB's 2-phase (temp-storage) API into the new-style
// 7-arg CUB API that merge.h calls:
//   MergeKeys(in1, count1, in2, count2, out, comp, policy)
// The policy provides the stream (or falls back to hipStreamPerThread).
// NOTE(HIP): hipCUB's MergeKeys uses int for num_keys; we cast the count.
struct DeviceMerge
{
  template <class _InIt1, class _OffsetT1, class _InIt2, class _OffsetT2,
            class _OutIt, class _Compare, class _Policy>
  static hipError_t MergeKeys(_InIt1 __in1, _OffsetT1 __count1,
                               _InIt2 __in2, _OffsetT2 __count2,
                               _OutIt __out, _Compare __comp, const _Policy& __policy)
  {
    ::cuda::stream_ref __sref =
      ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{hipStreamPerThread}, __policy);
    hipStream_t __stream = __sref.get();

    const int __n1 = static_cast<int>(__count1);
    const int __n2 = static_cast<int>(__count2);

    size_t __temp_bytes = 0;
    hipError_t __err    = ::hipcub::DeviceMerge::MergeKeys(
      static_cast<void*>(nullptr), __temp_bytes, __in1, __n1, __in2, __n2, __out, __comp, __stream);
    if (__err != hipSuccess)
    {
      return __err;
    }

    void* __d_temp = nullptr;
    if (__temp_bytes > 0)
    {
      __err = ::hipMalloc(&__d_temp, __temp_bytes);
      if (__err != hipSuccess)
      {
        return __err;
      }
    }
    __err = ::hipcub::DeviceMerge::MergeKeys(__d_temp, __temp_bytes, __in1, __n1, __in2, __n2, __out, __comp, __stream);
    if (__d_temp)
    {
      (void) ::hipFree(__d_temp);
    }
    return __err;
  }
};

// DevicePartition: HIP shim for the PSTL partition family.
//
// The PSTL backends call DevicePartition in two distinct modes:
//
//   A) Single-output (partition, stable_partition):
//        DevicePartition::If(temp, bytes, in, out, num_sel, n, pred, stream)
//      hipCUB::DevicePartition::If matches this exactly.
//
//   B) Two-output (partition_copy, rotate, rotate_copy):
//        DevicePartition::If / Flagged with out = partition_distinct_output_t<Sel,Rej>
//      CUB-internal type that splits selected/rejected elements into two separate output
//      iterators. hipCUB does not provide this; rocprim::partition_two_way does.
//
// Strategy: define cub::detail::select::partition_distinct_output_t here as a thin
// host/device holder. Then in DevicePartition provide two If/Flagged overloads:
//   - the more-specific template taking partition_distinct_output_t<Sel,Rej> explicitly
//     (preferred by overload resolution) -> rocprim::partition_two_way
//   - the generic _OutIt overload -> hipcub::DevicePartition::If / Flagged
// C++ overload resolution picks the more-specialised template, so the dispatch is correct.

namespace detail
{
namespace select
{

// Minimal replica of the CUB-internal type used by partition_copy.h, rotate.h, and
// rotate_copy.h.  On HIP we only need the constructor + data members; no CUB-internal
// kernel machinery is pulled in.
template <class _SelectedOutIt, class _RejectedOutIt>
struct partition_distinct_output_t
{
  _SelectedOutIt __selected_;
  _RejectedOutIt __rejected_;

  _CCCL_HOST_DEVICE partition_distinct_output_t(_SelectedOutIt __sel, _RejectedOutIt __rej)
      : __selected_(__sel)
      , __rejected_(__rej)
  {}
};

} // namespace select
} // namespace detail

// DevicePartition: provides If and Flagged with both single-output and two-output forms.
// The two-output form (taking partition_distinct_output_t) is a MORE SPECIFIC template
// instantiation than the generic _OutIt form; C++ overload resolution selects it first.
struct DevicePartition
{
  // --- If: two-output form (partition_copy path) ---
  // More specific than the _OutIt template below; selected when the output is the wrapper.
  template <class _InIt, class _SelOutIt, class _RejOutIt,
            class _NumSelIt, class _OffsetT, class _Pred>
  static hipError_t If(
    void* __d_temp, size_t& __bytes,
    _InIt __in,
    ::cub::detail::select::partition_distinct_output_t<_SelOutIt, _RejOutIt> __out,
    _NumSelIt __num_sel, _OffsetT __n, _Pred __pred, hipStream_t __stream)
  {
    return ::rocprim::partition_two_way(
      __d_temp, __bytes, __in,
      __out.__selected_, __out.__rejected_,
      __num_sel, static_cast<size_t>(__n), __pred, __stream);
  }

  // --- If: single-output form (partition / stable_partition path) ---
  template <class _InIt, class _OutIt, class _NumSelIt, class _OffsetT, class _Pred>
  static hipError_t If(
    void* __d_temp, size_t& __bytes, _InIt __in, _OutIt __out,
    _NumSelIt __num_sel, _OffsetT __n, _Pred __pred, hipStream_t __stream)
  {
    return ::hipcub::DevicePartition::If(
      __d_temp, __bytes, __in, __out, __num_sel, static_cast<_OffsetT>(__n), __pred, __stream);
  }

  // --- Flagged: two-output form (rotate / rotate_copy path) ---
  template <class _InIt, class _FlagIt, class _SelOutIt, class _RejOutIt,
            class _NumSelIt, class _OffsetT>
  static hipError_t Flagged(
    void* __d_temp, size_t& __bytes,
    _InIt __in, _FlagIt __flags,
    ::cub::detail::select::partition_distinct_output_t<_SelOutIt, _RejOutIt> __out,
    _NumSelIt __num_sel, _OffsetT __n, hipStream_t __stream)
  {
    return ::rocprim::partition_two_way(
      __d_temp, __bytes, __in, __flags,
      __out.__selected_, __out.__rejected_,
      __num_sel, static_cast<size_t>(__n), __stream);
  }

  // --- Flagged: single-output form ---
  template <class _InIt, class _FlagIt, class _OutIt, class _NumSelIt, class _OffsetT>
  static hipError_t Flagged(
    void* __d_temp, size_t& __bytes, _InIt __in, _FlagIt __flags, _OutIt __out,
    _NumSelIt __num_sel, _OffsetT __n, hipStream_t __stream)
  {
    return ::hipcub::DevicePartition::Flagged(
      __d_temp, __bytes, __in, __flags, __out, __num_sel, static_cast<_OffsetT>(__n), __stream);
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

#endif // _CCCL_HIP_COMPILATION() && !defined(_CCCL_COMPILER_HIPRTC)

#endif // _CUDA_STD___PSTL_CUDA_HIPCUB_H
