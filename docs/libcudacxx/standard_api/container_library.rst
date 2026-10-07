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
  :description: Documents the container library headers available in libhipcxx, including array, inplace_vector, mdspan, and span.
  :keywords: libhipcxx, ROCm, HIP, C++, containers, array, mdspan, span, inplace_vector

.. _libcudacxx-standard-api-container:

Container Library
=================

This page covers the container library headers available in libhipcxx, including array, inplace_vector, mdspan, and span.

.. toctree::
   :hidden:
   :maxdepth: 1

   container_library/array
   container_library/inplace_vector
   container_library/mdspan
   container_library/span

Any Standard C++ header not listed below is omitted.

.. list-table::
   :widths: 25 45 30 20
   :header-rows: 1

   * - **Header**
     - **Content**
     - **libhipcxx Availability**
     - **C++ Reference**

   * - :ref:`\<cuda/std/array\> <libcudacxx-standard-api-container-array>`
     - Fixed size array
     - libhipcxx 2.7
     - `\<array\> <https://en.cppreference.com/w/cpp/header/array>`_

   * - :ref:`\<cuda/std/inplace_vector\> <libcudacxx-standard-api-container-inplace-vector>`
     - Flexible size container with fixed capacity
     - libhipcxx 2.7
     - `\<inplace_vector\> <https://en.cppreference.com/w/cpp/header/inplace_vector>`_

   * - :ref:`\<cuda/std/mdspan\> <libcudacxx-standard-api-container-mdspan>`
     - Non-owning view into a multidimensional contiguous sequence of objects
     - libhipcxx 2.7
     - `\<mdspan\> <https://en.cppreference.com/w/cpp/header/mdspan>`_

   * - :ref:`\<cuda/std/span\> <libcudacxx-standard-api-container-span>`
     - Non-owning view into a contiguous sequence of objects
     - libhipcxx 2.7
     - `\<span\> <https://en.cppreference.com/w/cpp/header/span>`_
