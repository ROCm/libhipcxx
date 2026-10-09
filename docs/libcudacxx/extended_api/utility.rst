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
  :description: Overview of the libhipcxx utility extended API for HIP, including cuda::in_range and cuda::static_for.
  :keywords: libhipcxx, ROCm, HIP, C++, utility, in_range, static_for, compile-time loop

.. _libcudacxx-extended-api-utility:

Utility
========

This page covers the utility extended API.

.. toctree::
   :hidden:
   :maxdepth: 1

   cuda::in_range <utility/in_range>
   cuda::static_for <utility/static_for>

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **Since**

   * - :ref:`in_range <libcudacxx-extended-api-utility-in-range>`
     - Check if a value is within a range
     - libhipcxx 3.4

   * - :ref:`static_for <libcudacxx-extended-api-utility-static-for>`
     - Compile-time ``for`` loop
     - libhipcxx 3.4
