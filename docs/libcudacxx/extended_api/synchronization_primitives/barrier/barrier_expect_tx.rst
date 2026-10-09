..
    MIT License

    Copyright (c) 2024-2026 Advanced Micro Devices, Inc.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

.. meta::
  :description: API reference for cuda::device::barrier_expect_tx, which increments the expected transaction count of a shared-memory barrier in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, barrier_expect_tx, barrier, transaction count, shared memory, Hopper, mbarrier

.. _libcudacxx-extended-api-synchronization-barrier-barrier-expect-tx:

``cuda::device::barrier_expect_tx``
===================================

This page documents ``cuda::device::barrier_expect_tx``, which increments the expected transaction count of a shared memory barrier without signaling an arrival.

Defined in header ``<cuda/barrier>``:

.. code:: cpp

   __device__
   void cuda::device::barrier_expect_tx(
     cuda::barrier<cuda::thread_scope_block>& bar,
     ptrdiff_t transaction_count_update);

Increments the expected transaction count of a barrier in shared memory.

Preconditions
-------------

-  ``cuda::device::is_object_from(bar, cuda::device::address_space::shared) == true``
-  ``0 <= transaction_count_update && transaction_count_update <= (1 << 20) - 1``

Effects
-------

-  This function increments the expected transaction count by ``transaction_count_update``.
-  This function executes atomically.

..
   NVIDIA-specific (compute capability), not applicable to AMD GPUs.

   Notes
   -----

   This function can only be used under CUDA Compute Capability 9.0 (Hopper) or higher.

Example
-------

.. code:: cpp

   #include <cuda/barrier>
   #include <cuda/std/utility> // cuda::std::move

   #if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
   static_assert(false, "Insufficient CUDA Compute Capability: cuda::device::memcpy_expect_tx is not available.");
   #endif // __CUDA_MINIMUM_ARCH__

   __device__ alignas(16) int gmem_x[2048];

   __global__ void example_kernel() {
       using barrier_t = cuda::barrier<cuda::thread_scope_block>;
     alignas(16) __shared__ int smem_x[1024];
     __shared__ barrier_t bar;

     if (threadIdx.x == 0) {
       init(&bar, blockDim.x);
     }
     __syncthreads();

     if (threadIdx.x == 0) {
       cuda::device::memcpy_async_tx(smem_x, gmem_x, cuda::aligned_size_t<16>(sizeof(smem_x)), bar);
       cuda::device::barrier_expect_tx(bar, sizeof(smem_x));
     }
     auto token = bar.arrive(1);

     bar.wait(cuda::std::move(token));

     // smem_x contains the contents of gmem_x[0], ..., gmem_x[1023]
     smem_x[threadIdx.x] += 1;
   }

..
   `See it on Godbolt <https://godbolt.org/z/9Yj89P76z>`_
