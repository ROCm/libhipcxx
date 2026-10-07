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
  :description: Overview of the libhipcxx warp extended API, including warp shuffle operations for exchanging data between threads within a warp in HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, warp, warp_shuffle, warp_shuffle_idx, warp_shuffle_up, warp_shuffle_down, warp_shuffle_xor

.. _libcudacxx-extended-api-warp:

Warp
====

This page covers the warp extended API, providing generalized warp shuffle operations for exchanging data of arbitrary size between threads within a warp.

.. toctree::
   :hidden:
   :maxdepth: 1

   cuda::device::warp_shuffle <warp/warp_shuffle>

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **libhipcxx Availability**

   * - :ref:`warp_shuffle_idx <libcudacxx-extended-api-warp-warp-shuffle>`
     - Warp shuffle from a specific lane
     - libhipcxx 3.0

   * - :ref:`warp_shuffle_up <libcudacxx-extended-api-warp-warp-shuffle>`
     - Warp shuffle from original lane index - delta
     - libhipcxx 3.0

   * - :ref:`warp_shuffle_down <libcudacxx-extended-api-warp-warp-shuffle>`
     - Warp shuffle from original lane index + delta
     - libhipcxx 3.0

   * - :ref:`warp_shuffle_xor <libcudacxx-extended-api-warp-warp-shuffle>`
     - Warp shuffle from original lane index xor mask
     - libhipcxx 3.0
