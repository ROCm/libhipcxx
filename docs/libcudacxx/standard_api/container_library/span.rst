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
  :description: Documents cuda::std::span in libhipcxx, a non-owning contiguous sequence view with constexpr support from C++14 onwards and device-compatible range constructors.
  :keywords: libhipcxx, ROCm, HIP, C++, span, contiguous, view, constexpr, range

.. _libcudacxx-standard-api-container-span:

``<cuda/std/span>``
===================

This page documents ``cuda::std::span`` in libhipcxx, a non-owning view over a contiguous sequence of objects with full constexpr support from C++14 onwards.

Extensions
----------

- All features of ``<span>`` are made available in C++14 onwards.
- All features of ``<span>`` are made constexpr in C++14 onwards.

Restrictions
------------

- On device no exceptions are thrown in case of a bad access.
- The range-based constructors are emulated but not 100% equivalent.
