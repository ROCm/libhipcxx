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
  :description: Documents cuda::std::expected in libhipcxx, an optional value type with an error channel available from C++14 onwards with no device-side exceptions on bad access.
  :keywords: libhipcxx, ROCm, HIP, C++, expected, error handling, optional, value type

.. _libcudacxx-standard-api-utility-expected:

``<cuda/std/expected>``
=======================

This page documents ``cuda::std::expected`` in libhipcxx, a vocabulary type for error handling that holds either a value or an error, available from C++14 onwards.

See the documentation of the standard header `\<expected\> <https://en.cppreference.com/w/cpp/header/expected>`_

Extensions
----------

All features are available from C++14 onwards.

Restrictions
------------

On device no exceptions are thrown in case of a bad access.
