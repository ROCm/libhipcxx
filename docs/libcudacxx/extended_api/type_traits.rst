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
  :description: Overview of the libhipcxx type traits extended API for HIP, including floating-point, trivially copyable, bitwise comparable, and vector type traits.
  :keywords: libhipcxx, ROCm, HIP, C++, type traits, is_floating_point, is_trivially_copyable, is_bitwise_comparable, vector types

.. _libcudacxx-extended-api-type_traits:

Type traits
-----------

This page covers the type traits extended API.

.. toctree::
   :hidden:
   :maxdepth: 1

   type_traits/is_floating_point
   type_traits/is_trivially_copyable
   type_traits/is_bitwise_comparable
   type_traits/vector_types

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **Since**

   * - :ref:`cuda::is_floating_point <libcudacxx-extended-api-type_traits-is_floating_point>`
     - Tells whether a type is a floating point type
     - libhipcxx 3.0

   * - :ref:`Vector Type Traits <libcudacxx-extended-api-type_traits-vector_types>`
     - Type traits for HIP vector types
     - libhipcxx 3.4

   * - :ref:`cuda::is_trivially_copyable <libcudacxx-extended-api-type_traits-is_trivially_copyable>`
     - Relaxed trivially copyable check including extended floating-point types
     - libhipcxx 3.4

   * - :ref:`cuda::is_bitwise_comparable <libcudacxx-extended-api-type_traits-is_bitwise_comparable>`
     - User-specializable bitwise comparability check
     - libhipcxx 3.4
