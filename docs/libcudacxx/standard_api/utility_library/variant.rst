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
  :description: Documents cuda::std::variant in libhipcxx, a type-safe union type available from C++14 with constexpr support and a recursion-based cuda::std::visit implementation.
  :keywords: libhipcxx, ROCm, HIP, C++, variant, type-safe union, constexpr, visit, cuda::std::visit

.. _libcudacxx-standard-api-utility-variant:

``<cuda/std/variant>``
======================

This page documents ``cuda::std::variant`` in libhipcxx, a type-safe union that holds one of a set of types, available from C++14 onwards with constexpr support.

See the documentation of the standard header `\<variant\> <https://en.cppreference.com/w/cpp/header/variant>`_

Extensions
----------

- All features are available from C++14 onwards.
- All features are available at compile time if the different value types support it.

Restrictions
------------

On device no exceptions are thrown in case of a bad access.

Implementation notes
--------------------

``cuda::std::visit`` uses recursion instead of the usual array of function pointers. This improves runtime behavior
at the cost of longer compile times.
