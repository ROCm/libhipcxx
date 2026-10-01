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
  :description: API reference for cuda::pipeline::producer_acquire, which blocks a producer thread until the next pipeline stage becomes available in libhipcxx for HIP and CUDA.
  :keywords: libhipcxx, ROCm, HIP, C++, producer_acquire, pipeline, producer, blocking, stage

.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-producer-acquire:

``cuda::pipeline::producer_acquire``
====================================

This page documents ``cuda::pipeline::producer_acquire``, which blocks the current producer thread until the next pipeline stage is available.

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::thread_scope Scope>
   __host__ __device__
   void cuda::pipeline<Scope>::producer_acquire();

Blocks the current thread until the next *pipeline stage* is available.

.. note::

   - If the calling thread is a *consumer thread*, the behavior is undefined.
   - If the pipeline is in a :ref:`quitted state <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`,
     the behavior is undefined.
