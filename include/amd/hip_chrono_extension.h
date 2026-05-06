//===----------------------------------------------------------------------===//
//
// Modifications Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//===----------------------------------------------------------------------===//

#ifndef _AMD_HIP_CHRONO_EXTENSION_H
#define _AMD_HIP_CHRONO_EXTENSION_H

/**
 * For C++20, the standard requires that chrono::system_clock yields a UNIX
 * timestamp (see https://en.cppreference.com/w/cpp/chrono/system_clock).
 * There is currently no UNIX timestamp counter available on AMD hardware.
 * This header file implements a workaround for C++20.
 *
 * The idea is to send an initial host timestamp to the device and to store
 * it in __constant__ memory, then derive subsequent device-side timestamps
 * by adding the wall_clock64() delta since initialisation.
 *
 * IMPORTANT: This is an EXPERIMENTAL workaround.
 * IMPORTANT: Any application requiring a C++20 conforming system clock (i.e.
 * with UNIX timestamp epoch) needs to enable the workaround according to the
 * following steps:
 *
 *   1) The compile flag _LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP needs to be set
 *      (-D_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP).
 *   2) The linker flag -fgpu-rdc must be set.
 *   3) The macro LIBCUDACXX_HIP_DEFINE_SYSCLOCK_VARS needs to be invoked at
 *      file scope in a single translation unit, usually next to main(). The
 *      header <cuda/std/chrono> must be included to make this macro available.
 *   4) cuda::std::chrono::hip_gpu_ext::initialize_amdgpu_sysclock_on_current_device()
 *      or initialize_amdgpu_sysclock_on_device(int) needs to be called once
 *      on the host to initialise the offsets for a given device.
 *   5) Subsequent calls to cuda::std::chrono::system_clock::now() then
 *      return time_points starting at UNIX time.
 *
 * Example (full source in examples/hip/chrono_sysclock_workaround.hip):
 *
 *   #include <cuda/std/chrono>
 *
 *   // The macro defines the __constant__ state; one TU only, file scope.
 *   LIBCUDACXX_HIP_DEFINE_SYSCLOCK_VARS
 *
 *   int main()
 *   {
 *     cuda::std::chrono::hip_gpu_ext::initialize_amdgpu_sysclock_on_current_device();
 *     someKernelUsingSysclock<<<1, 1>>>();
 *   }
 */

#include <hip/hip_runtime.h>

#include <cassert>
#include <chrono>

// LIBCUDACXX_HIP_DEFINE_SYSCLOCK_VARS expands to definitions of the
// __constant__ variables declared inside cuda::std::chrono::hip_gpu_ext below.
// The qualified-declarator form requires the variables be already declared
// in that namespace, which they are by the extern declarations further down.
#define LIBCUDACXX_HIP_DEFINE_SYSCLOCK_VARS                                                    \
  __constant__ long long cuda::std::chrono::hip_gpu_ext::__unix_sysclock0_host_ticks = -1;     \
  __constant__ long long cuda::std::chrono::hip_gpu_ext::__offset_devclock0          = -1;

#define LIBCUDACXX_HIP_CHECK(command)         \
  {                                           \
    __hip_error = command;                    \
    assert(__hip_error == cudaSuccess);       \
  }

namespace cuda
{
namespace std
{
namespace chrono
{
namespace hip_gpu_ext
{

// __constant__ state populated by initialize_amdgpu_sysclock_on_*_device():
//   __unix_sysclock0_host_ticks - host-side UNIX timestamp at init, in
//                                  std::chrono::system_clock::period ticks
//   __offset_devclock0          - device wall_clock64() value at init
extern __constant__ long long __unix_sysclock0_host_ticks;
extern __constant__ long long __offset_devclock0;

// Sample wall_clock64() on the device and write to a single output cell.
// FIXME(HIP/AMD): kernel-launch latency means the host-side and device-side
// timestamps captured around this kernel are not perfectly synchronised.
__global__ inline void get_sysclock_offset_kernel(long long* __d_dev_sysclock_offset)
{
  *__d_dev_sysclock_offset = wall_clock64();
}

inline cudaError_t initialize_amdgpu_sysclock_on_current_device() noexcept
{
  cudaError_t __hip_error;

  long long* __d_dev_sysclock_offset;
  LIBCUDACXX_HIP_CHECK(cudaMalloc(&__d_dev_sysclock_offset, sizeof(long long)));

  // Get device sysclock offset and host UNIX timestamp at approximately the
  // same point in wall time.
  get_sysclock_offset_kernel<<<1, 1>>>(__d_dev_sysclock_offset);

  LIBCUDACXX_HIP_CHECK(cudaGetLastError());

  long long __h_host_unix_sysclock_ticks_elapsed =
    ::std::chrono::system_clock::now().time_since_epoch().count();

  long long __h_dev_sysclock_offset = -1;
  LIBCUDACXX_HIP_CHECK(
    cudaMemcpy(&__h_dev_sysclock_offset, __d_dev_sysclock_offset, sizeof(long long), cudaMemcpyDeviceToHost));

  LIBCUDACXX_HIP_CHECK(cudaMemcpyToSymbol(
    HIP_SYMBOL(__unix_sysclock0_host_ticks), &__h_host_unix_sysclock_ticks_elapsed, sizeof(long long)));
  LIBCUDACXX_HIP_CHECK(
    cudaMemcpyToSymbol(HIP_SYMBOL(__offset_devclock0), &__h_dev_sysclock_offset, sizeof(long long)));

  LIBCUDACXX_HIP_CHECK(cudaFree(__d_dev_sysclock_offset));
  return __hip_error;
}

inline cudaError_t initialize_amdgpu_sysclock_on_device(int __device_id) noexcept
{
  cudaError_t __hip_error;
  int __current_device_id;
  LIBCUDACXX_HIP_CHECK(cudaGetDevice(&__current_device_id));
  LIBCUDACXX_HIP_CHECK(cudaSetDevice(__device_id));

  LIBCUDACXX_HIP_CHECK(initialize_amdgpu_sysclock_on_current_device());

  LIBCUDACXX_HIP_CHECK(cudaDeviceSynchronize());
  LIBCUDACXX_HIP_CHECK(cudaSetDevice(__current_device_id));
  return __hip_error;
}

} // namespace hip_gpu_ext
} // namespace chrono
} // namespace std
} // namespace cuda

#endif // _AMD_HIP_CHRONO_EXTENSION_H
