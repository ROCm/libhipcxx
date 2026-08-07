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

// UNSUPPORTED: nvrtc, hiprtc
// UNSUPPORTED: enable-tile
// error: accessing gridDim/blockDim/blockIdx/threadIdx/warpSize is unsupported in tile code

// Anchors the *compile-time* warp extent baked into the hierarchy queries
// against the hardware, using two oracles that are independent of the library's
// own wave-size macro:
//
//   * the device-pass runtime `warpSize`, and
//   * the host-side `cudaDevAttrWarpSize` device attribute.
//
// The other hierarchy tests derive their expectation from _CCCL_HIP_WAVE_SIZE,
// which is the same macro the library uses to produce the value under test, so
// expectation and subject move together and the assertion holds whatever the
// macro answers. That makes them structurally unable to detect a wrong wave
// size (see the warp-level queries in cuda/__hierarchy/queries/extents.h, which
// are _CCCL_DEVICE_API on HIP precisely because the macro is meaningless in the
// host pass). This test compares against something the macro does not feed.
//
// It is backend-neutral: on NVIDIA both oracles report 32, which is what the
// queries answer there.

#include <cuda/hierarchy>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>

#include "test_macros.h"

TEST_DEVICE_FUNC void test_wave_size_against_runtime(unsigned* out)
{
  const auto hier = cuda::hierarchy{cuda::gpu_thread, cuda::grid_dims(dim3{gridDim}), cuda::block_dims(dim3{blockDim})};

  // The oracle: the wavefront size the device reports at run time.
  const unsigned rt_wave_size = static_cast<unsigned>(warpSize);

  // 1. thread-in-warp. The extent lives in the return type, so it must still be
  //    a compile-time constant, and that constant must be the hardware value.
  using warp_extents_t = decltype(cuda::gpu_thread.extents(cuda::warp, hier));
  static_assert(warp_extents_t::rank() == 1);
  static_assert(warp_extents_t::static_extent(0) != cuda::std::dynamic_extent,
                "the thread-in-warp extent must stay a compile-time constant");
  assert(warp_extents_t::static_extent(0) == rt_wave_size);

  assert(cuda::gpu_thread.extents(cuda::warp, hier).extent(0) == rt_wave_size);
  assert(cuda::gpu_thread.count(cuda::warp, hier) == rt_wave_size);
  assert(cuda::gpu_thread.static_count(cuda::warp, hier) == rt_wave_size);

  // 2. warp-in-block. The derived warp count must be computed against the same
  //    wavefront size; a partial trailing wave still counts as one.
  const unsigned threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  assert(cuda::warp.count(cuda::block, hier) == (threads_per_block + rt_wave_size - 1) / rt_wave_size);

  // Hand the compile-time value back so the host can cross-check it against
  // cudaDevAttrWarpSize as well -- that additionally catches a mismatch between
  // the architecture the kernel was compiled for and the one it runs on.
  if (out != nullptr && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0
      && blockIdx.z == 0)
  {
    *out = static_cast<unsigned>(warp_extents_t::static_extent(0));
  }
}

__global__ void wave_size_probe_kernel(unsigned* out)
{
  test_wave_size_against_runtime(out);
}

void test()
{
  int hw_wave_size = 0;
  assert(cudaDeviceGetAttribute(&hw_wave_size, cudaDevAttrWarpSize, 0) == cudaSuccess);
  assert(hw_wave_size == 32 || hw_wave_size == 64);

  unsigned* d_out = nullptr;
  assert(cudaMalloc(&d_out, sizeof(unsigned)) == cudaSuccess);

  // Include a block size that is not a whole multiple of either wave size, so
  // the warp-in-block rounding is exercised on both wave-32 and wave-64.
  const unsigned block_sizes[] = {32u, 64u, 96u, 128u};
  for (const unsigned block_size : block_sizes)
  {
    assert(cudaMemset(d_out, 0, sizeof(unsigned)) == cudaSuccess);

    wave_size_probe_kernel<<<2, block_size>>>(d_out);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    unsigned compiled_wave_size = 0;
    assert(cudaMemcpy(&compiled_wave_size, d_out, sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(compiled_wave_size == static_cast<unsigned>(hw_wave_size));
  }

  assert(cudaFree(d_out) == cudaSuccess);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
