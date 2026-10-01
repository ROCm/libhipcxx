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
  :description: Documents cuda::std::numeric in libhipcxx, providing constexpr numeric algorithms available from C++11 onwards, with notes on omitted parallel algorithms and saturation arithmetic.
  :keywords: libhipcxx, ROCm, HIP, C++, numeric algorithms, constexpr, nodiscard, saturation

.. _libcudacxx-standard-api-numerics-numeric:

``<cuda/std/numeric>``
======================

This page documents ``cuda::std::numeric`` in libhipcxx, which provides constexpr numeric algorithms including iota, accumulate, and inner_product, available from C++11 onwards.

Omissions
---------

- Currently we do not expose any parallel algorithms.
- Saturation arithmetics have not been implemented yet.

Extensions
----------

- All features of ``<numeric>`` are made available in C++11 onwards.
- All features of ``<numeric>`` are made constexpr in C++14 onwards.
- Algorithms that return a value and not an iterator have been marked ``[[nodiscard]]``.
