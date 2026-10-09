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
  :description: Documents cuda::std::optional in libhipcxx, an optional value type available from C++14, with constexpr support and support for optional references.
  :keywords: libhipcxx, ROCm, HIP, C++, optional, constexpr, value type, optional reference

.. _libcudacxx-standard-api-utility-optional:

``<cuda/std/optional>``
=======================

This page documents ``cuda::std::optional`` in libhipcxx, a vocabulary type that may or may not contain a value, available from C++14 onwards with constexpr support.

See the documentation of the standard header `\<optional\> <https://en.cppreference.com/w/cpp/header/optional>`_

Extensions
----------

- All features are available from C++14 onwards.
- All features are available at compile time if the value type supports it.

Restrictions
------------

On device no exceptions are thrown in case of a bad access.
