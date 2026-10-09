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
  :description: Overview of the libhipcxx numeric extended API, including narrowing casts and arithmetic with overflow checking and saturation for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, numeric, overflow, saturation, narrow, overflow_cast, overflow_result

.. _libcudacxx-extended-api-numeric:

Numeric
========

This page covers the numeric extended API, providing narrowing casts and arithmetic operations with overflow checking and saturation.

.. toctree::
   :hidden:
   :maxdepth: 1

   numeric/add_overflow
   numeric/saturating_add_overflow
   numeric/div_overflow
   numeric/saturating_div_overflow
   numeric/mul_overflow
   numeric/saturating_mul_overflow
   numeric/narrow
   numeric/overflow_cast
   numeric/overflow_result
   numeric/saturating_overflow_cast
   numeric/sub_overflow
   numeric/saturating_sub_overflow

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **Since**

   * - :ref:`cuda::narrow <libcudacxx-extended-api-numeric-narrow>`
     - Casts a value and checks whether the value has changed
     - libhipcxx 3.4

   * - :ref:`cuda::overflow_result <libcudacxx-extended-api-numeric-overflow_result>`
     - Represents the result of arithmetic operations that may overflow
     - libhipcxx 3.0

   * - :ref:`cuda::overflow_cast <libcudacxx-extended-api-numeric-overflow_cast>`
     - Casts a value with overflow checking
     - libhipcxx 3.0

   * - :ref:`cuda::add_overflow <libcudacxx-extended-api-numeric-add_overflow>`
     - Performs addition with overflow checking
     - libhipcxx 3.4

   * - :ref:`cuda::sub_overflow <libcudacxx-extended-api-numeric-sub_overflow>`
     - Performs subtraction with overflow checking
     - libhipcxx 3.4

   * - :ref:`cuda::div_overflow <libcudacxx-extended-api-numeric-div_overflow>`
     - Performs division with overflow checking
     - libhipcxx 3.4

   * - :ref:`cuda::mul_overflow <libcudacxx-extended-api-numeric-mul_overflow>`
     - Performs multiplication with overflow checking
     - libhipcxx 3.4

   * - :ref:`cuda::saturating_overflow_cast <libcudacxx-extended-api-numeric-saturating_overflow_cast>`
     - Performs saturating cast of a value with overflow checking
     - libhipcxx 3.4

   * - :ref:`cuda::saturating_add_overflow <libcudacxx-extended-api-numeric-saturating_add_overflow>`
     - Performs saturating addition with overflow checking
     - libhipcxx 3.4

   * - :ref:`cuda::saturating_sub_overflow <libcudacxx-extended-api-numeric-saturating_sub_overflow>`
     - Performs saturating subtraction with overflow checking
     - libhipcxx 3.4

   * - :ref:`cuda::saturating_div_overflow <libcudacxx-extended-api-numeric-saturating_div_overflow>`
     - Performs saturating division with overflow checking
     - libhipcxx 3.4

   * - :ref:`cuda::saturating_mul_overflow <libcudacxx-extended-api-numeric-saturating_mul_overflow>`
     - Performs saturating multiplication with overflow checking
     - libhipcxx 3.4
