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
  :description: Documents cuda::std::bitset in libhipcxx, a fixed-size sequence of bits with full constexpr support from C++14 and no device-side exceptions on bad access.
  :keywords: libhipcxx, ROCm, HIP, C++, bitset, fixed-size, constexpr, bit sequence

.. _libcudacxx-standard-api-utility-bitset:

``<cuda/std/bitset>``
======================

This page documents ``cuda::std::bitset`` in libhipcxx, which provides a fixed-size sequence of bits with full constexpr support from C++14 onwards.

Extensions
----------

All features of ``<bitset>`` are made constexpr in C++14 onwards.

Restrictions
------------

On device no exceptions are thrown in case of a bad access.
