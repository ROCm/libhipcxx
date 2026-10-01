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
  :description: Overview of the libhipcxx mdspan extended API, including restrict_accessor and restrict_mdspan for applying the restrict aliasing policy to multidimensional spans.
  :keywords: libhipcxx, ROCm, HIP, C++, mdspan, restrict_accessor, restrict_mdspan, aliasing, accessor

.. _libcudacxx-extended-api-mdspan:

Mdspan
======

This page covers the mdspan extended API, providing the restrict_accessor and restrict_mdspan types for applying the restrict aliasing policy to multidimensional spans.

.. toctree::
   :hidden:
   :maxdepth: 1

   mdspan/restrict_accessor

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`restrict mdspan and accessor <libcudacxx-extended-api-mdspan-restrict-accessor>`
     - ``mdspan`` and accessor with the *restrict* aliasing policy
     - CCCL 3.0.0
     - CUDA 13.0
