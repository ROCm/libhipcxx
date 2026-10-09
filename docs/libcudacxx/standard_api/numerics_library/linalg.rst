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
  :description: Documents cuda::std::linalg in libhipcxx, covering linear algebra accessors and transposed layouts including scaled, conjugated, transposed, and layout_transpose.
  :keywords: libhipcxx, ROCm, HIP, C++, linalg, linear algebra, transposed, accessor, conjugated

.. _libcudacxx-standard-api-numerics-linalg:

``<cuda/std/linalg>``
=====================

This page documents ``cuda::std::linalg`` in libhipcxx, providing linear algebra utilities including scaled, conjugated, and transposed accessors and views.

Provided functionalities
------------------------

- ``scaled()`` `std::linalg::scaled <https://en.cppreference.com/w/cpp/numeric/linalg/scaled>`_
- ``scaled_accessor`` `std::linalg::scaled_accessor <https://en.cppreference.com/w/cpp/numeric/linalg/scaled_accessor>`_
- ``conjugated()`` `std::linalg::conjugated <https://en.cppreference.com/w/cpp/numeric/linalg/conjugated>`_
- ``conjugated_accessor`` `std::linalg::conjugated_accessor <https://en.cppreference.com/w/cpp/numeric/linalg/conjugated_accessor>`_
- ``transposed()`` `std::linalg::transposed <https://en.cppreference.com/w/cpp/numeric/linalg/transposed>`_
- ``layout_transpose`` `std::linalg::layout_transpose <https://en.cppreference.com/w/cpp/numeric/linalg/layout_transpose>`_
- ``conjugate_transposed()`` `std::linalg::conjugate_transposed <https://en.cppreference.com/w/cpp/numeric/linalg/conjugate_transposed>`_

Extensions
----------

C++26 ``std::linalg`` accessors, transposed layout, and related functions are available in C++17.

Omissions
---------

Currently we do not expose any BLAS functions and layouts.

Restrictions
------------

- On device no exceptions are thrown in case of a bad access.
- MSVC is only supported with C++20.
