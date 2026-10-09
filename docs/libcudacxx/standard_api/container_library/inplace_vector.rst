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
  :description: Documents the cuda::std::inplace_vector container in libhipcxx, a fixed-capacity dynamically-sized container available from C++14 onwards.
  :keywords: libhipcxx, ROCm, HIP, C++, inplace_vector, fixed-capacity, container, ranges

.. _libcudacxx-standard-api-container-inplace-vector:

``<cuda/std/inplace_vector>``
==============================

This page documents ``cuda::std::inplace_vector`` in libhipcxx, a fixed-capacity, dynamically-sized contiguous container that stores its elements inline without heap allocation.

Extensions
----------

Most features of ``<inplace_vector>`` are made available in C++14 onwards.

Restrictions
------------

The range-based interface is only available with ranges support in C++17.
