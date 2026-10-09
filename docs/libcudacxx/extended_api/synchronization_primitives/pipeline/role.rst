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
  :description: API reference for cuda::pipeline_role, which specifies whether a thread is a producer or consumer in a partitioned pipeline in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, pipeline_role, producer, consumer, pipeline, partitioned

.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-role:

``cuda::pipeline_role``
=======================

This page documents ``cuda::pipeline_role``, an enumeration that specifies whether a thread acts as a producer or consumer in a partitioned pipeline.

Defined in header ``<cuda/pipeline>``:

.. code:: cpp

   enum class pipeline_role : /* unspecified */ {
     producer,
     consumer
   };

``cuda::pipeline_role`` specifies the role of a particular thread in a partitioned producer/consumer pipeline.

.. rubric:: Constants

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``cuda::pipeline_role::producer``
     - A producer thread that generates data and issuing
       :ref:`asynchronous operations <libcudacxx-extended-api-asynchronous-operations>`.
   * - ``cuda::pipeline_role::consumer``
     - A consumer thread that consumes data and waiting for previously
       :ref:`asynchronous operations <libcudacxx-extended-api-asynchronous-operations>` to complete.

.. rubric:: Example

See the :ref:`cuda::make_pipeline example <libcudacxx-extended-api-synchronization-pipeline-pipeline-make-pipeline-example>`.
