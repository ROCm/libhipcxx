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
  :description: API reference for cuda::atomic_thread_fence, a memory synchronization fence that establishes ordering of non-atomic and relaxed atomic accesses across a given thread scope in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, atomic_thread_fence, memory order, thread scope, fence, acquire, release

.. _libcudacxx-extended-api-synchronization-atomic-atomic-thread-fence:

``cuda::atomic::atomic_thread_fence``
=====================================

This page documents ``cuda::atomic_thread_fence``, which establishes memory ordering of non-atomic and relaxed atomic accesses across a specified thread scope.

Defined in header ``<cuda/atomic>``:

.. code:: cuda

   __host__ __device__
   void cuda::atomic_thread_fence(cuda::std::memory_order order,
                                  cuda::thread_scope scope = cuda::thread_scope_system);

Establishes memory synchronization ordering of non-atomic and relaxed atomic accesses, as instructed by ``order``,
for all threads within ``scope`` without an associated atomic operation. It has the same semantics as
`cuda::std::atomic_thread_fence <https://en.cppreference.com/w/cpp/atomic/atomic_thread_fence>`_.

Example
-------

The following code is an example of the :ref:`MessagePassing <libcudacxx-extended-api-memory-model-message-passing>` pattern:

.. code:: cuda

   #include <cstdio>
   #include <cuda/atomic>
   #include <cooperative_groups.h>

   namespace cg = cooperative_groups;

   __global__ void example_kernel(int* data, cuda::std::atomic_flag* flag) {
     assert(cg::grid_group::size() == 2);
     assert(cg::thread_block::size() == 1);

     if (blockIdx.x == 0) {
       *data = 42;
       cuda::atomic_thread_fence(cuda::memory_order_release,
                                 cuda::thread_scope_device);
       flag->test_and_set(cuda::std::memory_order_relaxed);
       flag->notify_one();
     }
     else {
       // an atomic operation is required to set up the synchronization
       flag->wait(false, cuda::std::memory_order_relaxed);
       cuda::atomic_thread_fence(cuda::memory_order_acquire,
                                 cuda::thread_scope_device);
       std::printf("%d\n", *data); // Prints 42
     }
   }

`See it on Godbolt <https://godbolt.org/z/aG37o5qxx>`_
