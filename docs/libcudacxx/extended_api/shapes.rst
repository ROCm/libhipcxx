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
  :description: Overview of the libhipcxx shapes API, including size_t and aligned_size_t types that define byte extents for asynchronous memory operations in HIP and CUDA.
  :keywords: libhipcxx, ROCm, HIP, C++, shapes, aligned_size_t, size_t, memory extent, alignment, memcpy_async


.. _libcudacxx-extended-api-memory-access-shapes:

Shapes
======

This page covers the shape types used to describe byte extents for asynchronous memory operations, including ``cuda::std::size_t`` and the alignment-aware ``cuda::aligned_size_t``.

.. toctree::
   :hidden:
   :maxdepth: 1

   shapes/aligned_size_t

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - `cuda::std::size_t <https://en.cppreference.com/w/cpp/types/size_t>`_
     - Defines an extent of bytes
     - libhipcxx 1.0.0 / CCCL 2.0.0 / CUDA 10.2
   * - :ref:`cuda::aligned_size_t <libcudacxx-extended-api-memory-access-shapes-aligned-size>`
     - Defines an extent of bytes with a statically defined alignment.
     - libhipcxx 1.2.0 / CCCL 2.0.0 / CUDA 11.1
