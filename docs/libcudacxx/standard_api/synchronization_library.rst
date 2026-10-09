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
  :description: Documents the synchronization library headers in libhipcxx, including atomic operations for heterogeneous GPU and CPU thread coordination.
  :keywords: libhipcxx, ROCm, HIP, C++, synchronization, atomic

.. _libcudacxx-standard-api-synchronization:

Synchronization Library
=======================

Any Standard C++ header not listed below is omitted.

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - `\<cuda/std/atomic\> <https://en.cppreference.com/w/cpp/header/atomic>`_
     - Atomic objects and operations. See also :ref:`Extended API <libcudacxx-extended-api-synchronization-atomic>`
     - libhipcxx 2.7

..
   Not supported in libhipcxx:
   <cuda/std/latch> - Single-phase asynchronous thread-coordination mechanism.
   <cuda/std/barrier> - Multi-phase asynchronous thread-coordination mechanism.
   <cuda/std/semaphore> - Primitives for constraining concurrent access.
