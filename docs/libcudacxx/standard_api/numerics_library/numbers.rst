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
  :description: Documents cuda::std::numbers in libhipcxx, providing mathematical constants such as pi and e in host and device code, with specializations for the HIP bfloat16 type.
  :keywords: libhipcxx, ROCm, HIP, C++, numbers, mathematical constants, bfloat16, floating point

.. _libcudacxx-standard-api-numerics-numbers:

``<cuda/std/numbers>``
======================

This page documents ``cuda::std::numbers`` in libhipcxx, which provides mathematical constants such as pi and e in
host and device code.

Extensions
----------

- The C++20 ``<numbers>`` mathematical constants are available in C++17.
- Specializations for the HIP extended floating-point type ``__hip_bfloat16`` are provided. Specializations for
  ``__half`` are currently not provided.
