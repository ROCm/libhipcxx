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

.. meta::
  :description: API reference for cuda::pipeline::~pipeline, the destructor that calls quit if not already called and destroys the pipeline object in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, pipeline destructor, pipeline, quit, destroy

.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-destructor:

``cuda::pipeline::~pipeline``
=============================

This page documents the ``cuda::pipeline`` destructor, which calls quit if not already called and destroys the pipeline object.

Defined in header ``<cuda/pipeline>``:

.. code:: cpp

   template <cuda::thread_scope Scope>
   __host__ __device__
   cuda::pipeline<Scope>::~pipeline();

Destructs the pipeline. Calls :ref:`cuda::pipeline::quit <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`
if it was not called by the current thread and destructs the pipeline.
