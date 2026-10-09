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
  :description: API reference for cuda::pipeline::producer_commit, which commits operations previously issued by the producer thread to the current pipeline stage in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, producer_commit, pipeline, producer, commit, stage

.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-producer-commit:

``cuda::pipeline::producer_commit``
===================================

This page documents ``cuda::pipeline::producer_commit``, which commits operations previously issued by the current producer thread to the current pipeline stage.

Defined in header ``<cuda/pipeline>``:

.. code:: cpp

   template <cuda::thread_scope Scope>
   __host__ __device__
   void cuda::pipeline<Scope>::producer_commit();

Commits operations previously issued by the current thread to the current *pipeline stage*.

.. note::

   - If the calling thread is a *consumer thread*, the behavior is undefined.
   - If the pipeline is in a :ref:`quitted state <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`,
     the behavior is undefined.
