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
  :description: API reference for cuda::pipeline_producer_commit, which binds pipeline operations to a barrier for completion notification in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, pipeline_producer_commit, pipeline, barrier, producer, commit

.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-pipeline-producer-commit:

``cuda::pipeline_producer_commit``
==================================

This page documents ``cuda::pipeline_producer_commit``, which commits all pending asynchronous operations from the current producer stage to the associated barrier.

Defined in header ``<cuda/pipeline>``:

.. code:: cpp

   template <cuda::thread_scope Scope>
   __host__ __device__
   void cuda::pipeline_producer_commit(cuda::pipeline<cuda::thread_scope_thread>& pipe,
                                       cuda::barrier<Scope>& bar);

Binds operations previously issued by the current thread to the named ``cuda::barrier`` such that a
``cuda::barrier::arrive`` is performed on completion. The bind operation implicitly increments the barrier's
current phase to account for the subsequent ``cuda::barrier::arrive``, resulting in a net change of 0.

.. rubric:: Parameters

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``pipe``
     - The thread-scoped ``cuda::pipeline`` object to wait on.
   * - ``bar``
     - The ``cuda::barrier`` to arrive on.

.. note::

   If the pipeline is in a :ref:`quitted state <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`,
   the behavior is undefined.

.. rubric:: Example

.. code:: cpp

   #include <cuda/pipeline>

   // Disables `barrier` initialization warning.
   #pragma nv_diag_suppress static_var_with_dynamic_init

   __global__ void
   example_kernel(cuda::std::uint64_t* global, cuda::std::size_t element_count) {
     extern __shared__ cuda::std::uint64_t shared[];
     __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

     init(&barrier, 1);
     cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

     pipe.producer_acquire();
     for (cuda::std::size_t i = 0; i < element_count; ++i)
       cuda::memcpy_async(shared + i, global + i, sizeof(*global), pipe);
     pipeline_producer_commit(pipe, barrier);
     barrier.arrive_and_wait();
     pipe.consumer_release();
   }

..
   `See it on Godbolt <https://godbolt.org/z/sGzKe9obf>`_
