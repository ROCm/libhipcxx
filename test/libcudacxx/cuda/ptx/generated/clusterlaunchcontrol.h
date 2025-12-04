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

// This file was automatically generated. Do not edit.

// We use a special strategy to force the generation of the PTX. This is mainly
// a fight against dead-code-elimination in the NVVM layer.
//
// The reason we need this strategy is because certain older versions of ptxas
// segfault when a non-sensical sequence of PTX is generated. So instead, we try
// to force the instantiation and compilation to PTX of all the overloads of the
// PTX wrapping functions.
//
// We do this by writing a function pointer of each overload to the kernel
// parameter `fn_ptr`.
//
// Because `fn_ptr` is possibly visible outside this translation unit, the
// compiler must compile all the functions which are stored.

__global__ void test_clusterlaunchcontrol(void** fn_ptr)
{
#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [addr], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(void*, cuda::std::uint64_t*)>(cuda::ptx::clusterlaunchcontrol_try_cancel));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128
        // [addr], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(void*, cuda::std::uint64_t*)>(cuda::ptx::clusterlaunchcontrol_try_cancel_multicast));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128
        // [addr], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(void*, cuda::std::uint64_t*)>(cuda::ptx::clusterlaunchcontrol_try_cancel_multicast));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(NV_PROVIDES_SM_100,
               (
                   // clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 pred_is_canceled, try_cancel_response;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(longlong2)>(cuda::ptx::clusterlaunchcontrol_query_cancel_is_canceled));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(NV_PROVIDES_SM_100,
               (
                   // clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 ret_dim, try_cancel_response;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::int32_t (*)(longlong2)>(
                     cuda::ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(NV_PROVIDES_SM_100,
               (
                   // clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 ret_dim, try_cancel_response;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::int32_t (*)(longlong2)>(
                     cuda::ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(NV_PROVIDES_SM_100,
               (
                   // clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 ret_dim, try_cancel_response;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::int32_t (*)(longlong2)>(
                     cuda::ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_z));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 block_dim, try_cancel_response;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&block_dim)[4], longlong2)>(
          cuda::ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid));));
#endif // __cccl_ptx_isa >= 860
}
