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
  :description: Overview of the libhipcxx math extended API, including ceiling division, rounding, and integer logarithm utilities for HIP.
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
   math/ipow
   math/pow2
   math/isqrt
   math/neg
   math/uabs
   math/fast_mod_div
   math/mul_hi

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **libhipcxx Availability**

   * - :ref:`ceil_div <libcudacxx-extended-api-math-ceil-div>`
     - Ceiling division
     - libhipcxx 2.7

   * - :ref:`round_up <libcudacxx-extended-api-math-round-up>`
     - Round up to the next multiple
     - libhipcxx 3.0

   * - :ref:`round_down <libcudacxx-extended-api-math-round-down>`
     - Round down to the previous multiple
     - libhipcxx 3.0

   * - :ref:`ilog2 <libcudacxx-extended-api-math-ilog>`
     - Integer logarithm to the base 2
     - libhipcxx 3.0

   * - :ref:`ilog10 <libcudacxx-extended-api-math-ilog>`
     - Integer logarithm to the base 10
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`ipow <libcudacxx-extended-api-math-ipow>`
     - Integer power
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`is_power_of_two <libcudacxx-extended-api-math-pow2>`
     - If the value is a power of two
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`isqrt <libcudacxx-extended-api-math-isqrt>`
     - Integer square root
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`neg <libcudacxx-extended-api-math-neg>`
     - Integer negation
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`next_power_of_two <libcudacxx-extended-api-math-pow2>`
     - Next power of two
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`prev_power_of_two <libcudacxx-extended-api-math-pow2>`
     - Previous power of two
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`uabs <libcudacxx-extended-api-math-uabs>`
     - Unsigned absolute value
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`fast_mod_div <libcudacxx-extended-api-math-fast-mod-div>`
     - Fast Modulo/Division
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`mul_hi <libcudacxx-extended-api-math-mul-hi>`
     - Most significant half of the product
     - CCCL 3.2.0
     - CUDA 13.2
