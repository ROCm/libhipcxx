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
  :description: Documents cuda::std::mdspan in libhipcxx, a non-owning multidimensional array view with extensions including aligned_accessor, dims, and debug bounds checking.
  :keywords: libhipcxx, ROCm, HIP, C++, mdspan, multidimensional, span, aligned_accessor, dims

.. _libcudacxx-standard-api-container-mdspan:

``<cuda/std/mdspan>``
=====================

This page documents ``cuda::std::mdspan`` in libhipcxx, a multidimensional span for non-owning views of contiguous data with C++17 availability and C++26 backports.

Provided functionalities
------------------------

- All features of ``<mdspan>`` are made available in C++17 onwards.
- C++26 ``std::dims`` is made available in C++17 onwards.
- C++26 ``std::aligned_accessor`` is made available in C++17 onwards.

Extensions
----------

- The C++23 multidimensional ``operator[]`` is replaced with ``operator()`` in previous C++ standards.
- Detection of out-of-bounds accesses is available in debug mode.

Restrictions
------------

On device no exceptions are thrown in case of a bad access.
