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
  :description: libhipcxx HIP-specific abstractions and namespaces
  :keywords: libhipcxx, ROCm, HIP, namespaces, hip aliasing, cuda, std, device

.. _libhipcxx-hip-abstractions:

********************************************************************
HIP-specific abstractions and namespaces
********************************************************************

Some abstractions that libhipcxx provides have no equivalent in the C++ Standard Library, but are
fundamental to the HIP C++ programming model. These are provided alongside the Standard Library
facilities and their extensions.

HIP aliasing
============

Instead of using ``cuda::`` and ``cuda::std::``, you can use ``hip::`` and ``hip::std::``. Both
include variants, through ``hip`` or ``cuda``, can be used interchangeably and resolve to the same
headers.

Choosing a namespace
====================

The following list summarizes which namespace to use and where the facilities in it can run:

* ``std::`` / ``<*>``: Your host compiler's Standard Library, which works in ``__host__`` code only.
  libhipcxx does not replace or interfere with the host compiler's Standard Library.
* ``cuda::std::`` / ``hip::std::`` / ``<cuda/std/*>`` / ``<hip/std/*>``: Conforming implementations
  of facilities from the Standard Library that work in ``__host__`` and ``__device__`` code.
* ``cuda::`` / ``hip::`` / ``<cuda/*>`` / ``<hip/*>``: Conforming extensions to the Standard Library
  that work in ``__host__`` and ``__device__`` code.
* ``cuda::device`` / ``hip::device`` / ``<cuda/device/*>`` / ``<hip/device/*>``: Conforming
  extensions to the Standard Library that work only in ``__device__`` code.

The following example shows the three variants side by side:

.. code-block:: cpp

    // Standard C++, __host__ only.
    #include <atomic>
    std::atomic<int> x;

    // HIP C++, __host__ __device__.
    // Strictly conforming to the C++ Standard.
    #include <cuda/std/atomic>
    cuda::std::atomic<int> x;

    // HIP C++, __host__ __device__.
    // Conforming extensions to the C++ Standard.
    #include <cuda/atomic>
    cuda::atomic<int, cuda::thread_scope_block> x;
