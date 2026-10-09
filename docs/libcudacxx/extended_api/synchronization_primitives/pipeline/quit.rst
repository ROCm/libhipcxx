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
  :description: API reference for cuda::pipeline::quit, which quits the current thread's participation in the pipeline and optionally releases collective ownership of the shared state in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, pipeline quit, pipeline, shared state, collective ownership, release

.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-quit:

``cuda::pipeline::quit``
========================

This page documents ``cuda::pipeline::quit``, which releases the calling thread's shared ownership of the pipeline and its shared state.

Defined in header ``<cuda/pipeline>``:

.. code:: cpp

   template <cuda::thread_scope Scope>
   __host__ __device__
   bool cuda::pipeline<Scope>::quit();

Quits the current thread's participation in the collective ownership of the corresponding
:ref:`cuda::pipeline_shared_state <libcudacxx-extended-api-synchronization-pipeline-pipeline-shared-state>`.
Ownership of :ref:`cuda::pipeline_shared_state <libcudacxx-extended-api-synchronization-pipeline-pipeline-shared-state>`
is released by the last invoking thread.

.. rubric:: Return Value

``true`` if ownership of the *shared state* was released, otherwise ``false``.

.. note::

   After the completion of a call to ``cuda::pipeline::quit``, no other operations other than
   :ref:`cuda::pipeline::~pipeline <libcudacxx-extended-api-synchronization-pipeline-pipeline-destructor>` may be
   called by the current thread.
