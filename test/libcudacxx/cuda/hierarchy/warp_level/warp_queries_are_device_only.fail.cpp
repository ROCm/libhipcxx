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

// REQUIRES: hipcc
// UNSUPPORTED: nvrtc, hiprtc
// UNSUPPORTED: enable-tile

// NOTE(HIP/AMD): pins the warp-level hierarchy queries as device-only on HIP.
//
// _CCCL_HIP_WAVE_SIZE is derived from the __GFX*__ predefines, which clang emits
// only in the device pass; in the host pass the macro silently takes its wave-32
// fallback. A host-callable warp-level query would therefore answer 32 on GFX9,
// so cuda/__hierarchy/queries/extents.h marks these _CCCL_DEVICE_API on HIP
// (upstream uses _CCCL_API, which is fine there because 32 is pass-invariant on
// NVIDIA). This is the same invariant established for the warp primitives: the
// wave size is device-only, anything host-visible must be pass-invariant.
//
// The companion wave_size_matches_hardware.pass.cpp anchors the *value* against
// the runtime wavefront size, but it runs device-side, so it stays green if the
// annotation regresses to _CCCL_API. Only this test pins the annotation itself.
// A SFINAE detector cannot substitute for it: the host/device restriction is a
// deferred diagnostic, so `decltype(gpu_thread.count(cuda::warp, hier))` is
// well-formed even where the call is an error.

#include <cuda/hierarchy>

using hierarchy_t =
  cuda::hierarchy<cuda::thread_level,
                  cuda::hierarchy_level_desc<cuda::grid_level, cuda::std::extents<unsigned, 2, 1, 1>>,
                  cuda::hierarchy_level_desc<cuda::block_level, cuda::std::extents<unsigned, 128, 1, 1>>>;

// Note that `main` is `__host__ __device__` under the HIP test harness, so the
// calls below are diagnosed in the host pass exactly as they would be in
// ordinary host code. hipcc runs clang -verify over *both* passes and every
// directive must be satisfied in each, so the calls and their expectations are
// confined to the host pass -- in the device pass they are correct code and
// produce nothing to match.
int main(int, char**)
{
  [[maybe_unused]] const hierarchy_t hier{cuda::gpu_thread, cuda::grid_dims<2>(), cuda::block_dims<128>()};

#if !defined(__HIP_DEVICE_COMPILE__)
  // The expectation is deliberately a short prefix of the diagnostic: the full text spells out the whole
  // hierarchy_t template-id, which does not fit the column limit and would churn on any unrelated rename.
  // expected-error@*:* {{reference to __device__ function '__call<unsigned int, cuda::hierarchy<}}
  [[maybe_unused]] const auto threads_per_warp = cuda::gpu_thread.count(cuda::warp, hier);

  // expected-error@*:* {{reference to __device__ function '__call<unsigned int, cuda::hierarchy<}}
  [[maybe_unused]] const auto warps_per_block = cuda::warp.count(cuda::block, hier);
#else // ^^^ host pass ^^^ / vvv device pass vvv
  // The same queries are correct code here, and must stay that way.
  // expected-no-diagnostics
  [[maybe_unused]] const auto threads_per_warp = cuda::gpu_thread.count(cuda::warp, hier);
  [[maybe_unused]] const auto warps_per_block  = cuda::warp.count(cuda::block, hier);
#endif // !__HIP_DEVICE_COMPILE__

  return 0;
}
