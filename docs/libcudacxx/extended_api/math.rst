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
  :description: Overview of the libhipcxx math extended API, including ceiling division, rounding, and integer logarithm utilities for HIP and CUDA.
  :keywords: libhipcxx, ROCm, HIP, C++, math, ceil_div, round_up, round_down, ilog2, ilog10

.. _libcudacxx-extended-api-math:

Math
====

This page covers the math extended API, providing integer arithmetic utilities including ceiling division, rounding, and integer logarithm functions.

.. toctree::
   :hidden:
   :maxdepth: 1

   math/ceil_div
   math/round_up
   math/round_down
   math/ilog

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`ceil_div <libcudacxx-extended-api-math-ceil-div>`
     - Ceiling division
     - CCCL 2.7.0
     - CUDA 12.8

   * - :ref:`round_up <libcudacxx-extended-api-math-round-up>`
     - Round up to the next multiple
     - CCCL 2.9.0
     - CUDA 12.9

   * - :ref:`round_down <libcudacxx-extended-api-math-round-down>`
     - Round down to the previous multiple
     - CCCL 2.9.0
     - CUDA 12.9

   * - :ref:`ilog2 <libcudacxx-extended-api-math-ilog>`
     - Integer logarithm to the base 2
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`ilog10 <libcudacxx-extended-api-math-ilog>`
     - Integer logarithm to the base 10
     - CCCL 3.0.0
     - CUDA 13.0
