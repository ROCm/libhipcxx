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
     - libhipcxx 1.0.0 / CCCL 2.0.0 / CUDA 10.2
   * - `\<cuda/std/latch\> <https://en.cppreference.com/w/cpp/header/latch>`_
     - Single-phase asynchronous thread-coordination mechanism. See also :ref:`Extended API <libcudacxx-extended-api-synchronization-latch>`
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0
   * - `\<cuda/std/barrier\> <https://en.cppreference.com/w/cpp/header/barrier>`_
     - Multi-phase asynchronous thread-coordination mechanism. See also :ref:`Extended API <libcudacxx-extended-api-synchronization-barrier>`
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0
   * - `\<cuda/std/semaphore\> <https://en.cppreference.com/w/cpp/header/semaphore>`_
     - Primitives for constraining concurrent access. See also :ref:`Extended API <libcudacxx-extended-api-synchronization-counting-semaphore>`
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0
