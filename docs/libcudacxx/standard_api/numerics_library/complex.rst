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
  :description: Documents cuda::std::complex in libhipcxx, including the omission of complex literals, the demotion of long double in device code, and extensions for infinity handling and half/bfloat16 support.
  :keywords: libhipcxx, ROCm, HIP, C++, complex numbers, half, bfloat16, constexpr

.. _libcudacxx-standard-api-numerics-complex:

``<cuda/std/complex>``
======================

This page documents the ``cuda::std::complex`` support in libhipcxx, including the omission of complex literals and extensions for infinity handling and half and bfloat16 types.

Omissions
---------

When a translation unit is compiled with hipcc, ``complex`` does not provide the complex literals ``i``, ``if``, and
``il``. Their suffixes do not start with an underscore, a form that is reserved for the standard library. The literals
are available when a translation unit is compiled directly with the host compiler.

``complex<long double>`` is available, but ``long double`` is demoted to ``double`` in device code. The type therefore
has a different size in host and device code and must not be shared between them.

Extensions
----------

- Handling of infinities

  Our implementation by default recovers infinite values during multiplication and division. The recovery only runs
  when the unrecovered result is ``NaN`` in both components, but it enlarges the generated code considerably, so we
  allow disabling that canonicalization if it is not desired.

  Definition of ``LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS`` disables canonicalization for both multiplication *and* division.

  Definition of ``LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_MULTIPLICATION`` or ``LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_DIVISION`` disables
  canonicalization for multiplication or division individually.

- Support for half and bfloat16

  Our implementation includes support for the ``__half`` type from ``<hip/hip_fp16.h>`` and the ``__hip_bfloat16``
  type from ``<hip/hip_bf16.h>``. Both are available when the translation unit is compiled with hipcc, including
  host-only translation units. They are not available when a translation unit is compiled directly with the host
  compiler.

- C++20 constexpr ``<complex>`` is available in C++17.
