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
  :description: API reference for cuda::pipeline::consumer_release, which releases the current pipeline stage for reuse by producer threads in libhipcxx for HIP and CUDA.
  :keywords: libhipcxx, ROCm, HIP, C++, consumer_release, pipeline, consumer, stage, release

.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-consumer-release:

``cuda::pipeline::consumer_release``
====================================

This page documents ``cuda::pipeline::consumer_release``, which releases the current pipeline stage for reuse by producer threads.

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::thread_scope Scope>
   __host__ __device__
   void cuda::pipeline<Scope>::consumer_release();

Releases the current *pipeline stage*.

.. note::

   - If the calling thread is a *producer thread*, the behavior is undefined.
   - If the pipeline is in a :ref:`quitted state <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`,
     the behavior is undefined.
