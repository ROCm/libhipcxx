..
    MIT License

    Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

.. _libcudacxx-extended-api-synchronization-barrier-barrier-arrive-tx:

cuda::device::barrier_arrive_tx
===================================

Defined in header ``<cuda/barrier>``:

.. code:: cuda

   __device__
   cuda::barrier<cuda::thread_scope_block>::arrival_token
   cuda::device::barrier_arrive_tx(
     cuda::barrier<cuda::thread_scope_block>& bar,
     ptrdiff_t arrive_count_update,
     ptrdiff_t transaction_count_update);

Arrives at a barrier in shared memory, decrementing the arrival count and incrementing the expected transaction count.

Preconditions
-------------

-  ``__isShared(&bar) == true``
-  ``1 <= arrive_count_update && transaction_count_update <= (1 << 20) - 1``
-  ``0 <= transaction_count_update && transaction_count_update <= (1 << 20) - 1``

Effects
-------

-  This function constructs an arrival_token object associated with the
   phase synchronization point for the current phase. Then, decrements
   the arrival count by ``arrive_count_update`` and increments the
   expected transaction count by ``transaction_count_update``.
-  This function executes atomically. The call to this function strongly
   happens-before the start of the phase completion step for the current
   phase.

Notes
-----

This function can only be used under CUDA Compute Capability 9.0 (Hopper) or higher.

To check if ``cuda::device::barrier_arrive_tx`` is available, use the ``__cccl_lib_local_barrier_arrive_tx``
feature flag, as shown in the example code below.

Return Value
------------

``cuda::device::barrier_arrive_tx`` returns the constructed ``arrival_token`` object.

Example
-------

Below example shows only ``cuda::device::barrier_arrive_tx``. A more extensive example can be found in the
:ref:`cuda::device::memcpy_async_tx <libcudacxx-extended-api-asynchronous-operations-memcpy-async-tx>` documentation.

.. code:: cuda

   #include <cuda/barrier>
   #include <cuda/std/utility> // cuda::std::move

   #ifndef  __cccl_lib_local_barrier_arrive_tx
   static_assert(false, "Insufficient libhipcxx version: cuda::device::arrive_tx is not yet available.");
   #endif // __cccl_lib_local_barrier_arrive_tx

   __global__ void example_kernel() {
     __shared__ cuda::barrier<cuda::thread_scope_block> bar;
     if (threadIdx.x == 0) {
       init(&bar, blockDim.x);
     }
     __syncthreads();

     auto token = cuda::device::barrier_arrive_tx(bar, 1, 0);

     bar.wait(cuda::std::move(token));
   }

`See it on Godbolt <https://godbolt.org/z/1vxcGrT8j>`_
