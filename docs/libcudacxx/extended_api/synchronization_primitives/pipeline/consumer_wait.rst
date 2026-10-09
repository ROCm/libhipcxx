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
  :description: API reference for cuda::pipeline::consumer_wait, consumer_wait_for, and consumer_wait_until, which block a consumer thread until pipeline stage operations complete in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, consumer_wait, pipeline, consumer, blocking, timeout, chrono

.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-consumer-wait:

``cuda::pipeline::consumer_wait``
=================================

This page documents ``cuda::pipeline::consumer_wait``, which blocks the current thread until all operations in the current pipeline stage complete.

Defined in header ``<cuda/pipeline>``:

.. code:: cpp

   // (1)
   template <cuda::thread_scope Scope>
   __host__ __device__
   void cuda::pipeline<Scope>::consumer_wait();

   // (2)
   template <cuda::thread_scope Scope>
   template <typename Rep, typename Period>
   __host__ __device__
   bool cuda::pipeline<Scope>::consumer_wait_for(
     cuda::std::chrono::duration<Rep, Period> const& duration);

   // (3)
   template <cuda::thread_scope Scope>
   template <typename Clock, typename Duration>
   __host__ __device__
   bool cuda::pipeline<Scope>::consumer_wait_until(
     cuda::std::chrono::time_point<Clock, Duration> const& time_point);

1. Blocks the current thread until all operations committed to the current *pipeline stage* complete.
2. Blocks the current thread until all operations committed to the current *pipeline stage* complete or after the
   specified timeout duration.
3. Blocks the current thread until all operations committed to the current *pipeline stage* complete or until specified
   time point has been reached.

.. rubric:: Parameters

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``duration``
     - An object of type ``cuda::std::chrono::duration`` representing the maximum time to spend waiting.
   * - ``time_point``
     - An object of type ``cuda::std::chrono::time_point`` representing the time when to stop waiting.


.. rubric:: Return Value

``false`` if the *wait* timed out, ``true`` otherwise.

.. note::

   - If the calling thread is a *producer thread*, the behavior is undefined.
   - If the pipeline is in a :ref:`quitted state <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`,
     the behavior is undefined.
