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

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <testing.cuh>

// Verifies that the device-side dynamic_shared_memory_option::size_bytes() reports
// the actual dynamic shared memory allocated for the launch. On HIP this exercises
// the HSA dispatch packet readback (cuda::__hip_group_segment_size() in
// <cuda/__device/hip_dynamic_smem.h>): group_segment_size minus the static LDS.
// On CUDA it reads %dynamic_smem_size. The kernel compares the device readback
// against the size configured on the host.
template <class T>
struct SizeBytesKernel
{
  // Independently computed on the host as n * sizeof(T) -- deliberately NOT
  // opt.size_bytes(). host size_bytes() and device size_bytes() are different code
  // paths (host returns the configured n * sizeof(T); device performs a hardware
  // readback -- on HIP the HSA dispatch packet group_segment_size minus static
  // LDS), so comparing the device readback against this value is a real check, not
  // a tautology, and it fails if the readback machinery is wrong.
  cuda::std::size_t expected_size_bytes;

  template <class Config>
  __device__ void operator()(const Config& config)
  {
    auto& opt =
      cuda::__detail::find_option_in_tuple<cuda::__detail::launch_option_kind::dynamic_shared_memory>(config.options());
    CCCLRT_REQUIRE_DEVICE(opt.size_bytes() == expected_size_bytes);
  }
};

template <class T>
void test_size_bytes(cuda::stream_ref stream, cuda::std::size_t n)
{
  // Expectation derived only from the launch parameters, not from size_bytes().
  const cuda::std::size_t expected_size_bytes = n * sizeof(T);

  auto opt = cuda::dynamic_shared_memory<T[]>(n);
  // Host path: size_bytes() must return the configured size.
  CCCLRT_REQUIRE(opt.size_bytes() == expected_size_bytes);

  const auto config = cuda::make_config(cuda::block_dims<1, 1>(), cuda::grid_dims<1, 1>(), opt);
  // Device path: the readback (checked inside the kernel) must match too.
  cuda::launch(stream, config, SizeBytesKernel<T>{expected_size_bytes});
  stream.sync();
}

C2H_TEST("Dynamic shared memory size_bytes readback", "[launch]")
{
  cuda::device_ref device = cuda::devices[0];
  cuda::stream stream{device};

  test_size_bytes<int>(stream, 1);
  test_size_bytes<int>(stream, 256);
  test_size_bytes<float>(stream, 128);
  test_size_bytes<double>(stream, 64);
  test_size_bytes<char>(stream, 1000);
}
