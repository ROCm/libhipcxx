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
  :description: API reference for cuda::pipeline_consumer_wait_prior, which blocks the current thread until all pipeline operations up to a prior stage complete in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, pipeline_consumer_wait_prior, pipeline, consumer, blocking, prior stage

.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-consumer-wait-prior:

``cuda::pipeline_consumer_wait_prior``
======================================

This page documents ``cuda::pipeline_consumer_wait_prior``, which waits for pipeline stages prior to a specified number of stages before the current stage.

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::std::uint8_t Prior>
   __host__ __device__
   void cuda::pipeline_consumer_wait_prior(cuda::pipeline<thread_scope_thread>& pipe);

Let *Stage* be the pipeline stage ``Prior`` stages before the current one (counting the current one).
Blocks the current thread until all operations committed to *pipeline stages* up to *Stage* complete.
All stages up to *Stage* (exclusive) are implicitly released.

.. rubric:: Template Parameters

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``Prior``
     - The index of the pipeline stage *Stage* (see above) counting up from the current one. The index of the current stage is ``0``.

.. rubric:: Parameters

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``pipe``
     - The thread-scoped ``cuda::pipeline`` object to wait on.

.. note::

   If the pipeline is in a :ref:`quitted state <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`,
   the behavior is undefined.

.. rubric:: Example

.. code:: cuda

   #include <cuda/pipeline>

   __global__ void example_kernel(uint64_t* global, cuda::std::size_t element_count) {
     extern __shared__ uint64_t shared[];

     cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
     for (cuda::std::size_t i = 0; i < element_count; ++i) {
       pipe.producer_acquire();
       cuda::memcpy_async(shared + i, global + i, sizeof(*global), pipe);
       pipe.producer_commit();
     }

     // Wait for operations committed in all stages but the last one.
     cuda::pipeline_consumer_wait_prior<1>(pipe);
     pipe.consumer_release();

     // Wait for operations committed in all stages.
     cuda::pipeline_consumer_wait_prior<0>(pipe);
     pipe.consumer_release();
   }

..
   `See it on Godbolt <https://godbolt.org/z/aT5hb84PY>`_
