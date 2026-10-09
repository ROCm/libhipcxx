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
  :description: Overview of the libhipcxx bit manipulation extended API, including bitmask generation, bit reversal, and bitfield insert and extract operations.
  :keywords: libhipcxx, ROCm, HIP, C++, bit manipulation, bitmask, bit_reverse, bitfield_insert, bitfield_extract

.. _libcudacxx-extended-api-bit:

Bit
===

This page covers the bit-manipulation extended API, providing utilities for bitmask generation, bit reversal, and bitfield insert and extract operations.

.. toctree::
   :hidden:
   :maxdepth: 1

   bit/bitmask
   bit/bit_reverse
   bit/bitfield_insert
   bit/bitfield_extract

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **Since**

   * - :ref:`bitmask <libcudacxx-extended-api-bit-bitmask>`
     - Generate a bitmask
     - libhipcxx 3.0

   * - :ref:`bit_reverse <libcudacxx-extended-api-bit-bit_reverse>`
     - Reverse the order of bits
     - libhipcxx 3.0

   * - :ref:`bitfield_insert <libcudacxx-extended-api-bit-bitfield_insert>`
     - Insert a bitfield
     - libhipcxx 3.0

   * - :ref:`bitfield_extract <libcudacxx-extended-api-bit-bitfield_extract>`
     - Extract a bitfield
     - libhipcxx 3.0
