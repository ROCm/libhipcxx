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

.. _libcudacxx-extended-api-memory-access-shapes:

Shapes
======

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
