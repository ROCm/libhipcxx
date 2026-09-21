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

.. _libcudacxx-standard-api-numerics-complex:

``<cuda/std/complex>``
======================

Omissions
---------

  When using libhipcxx with NVCC, ``complex`` does not support ``long double`` or ``complex`` literals (``_i``, ``_if``, and ``_il``).
  NVCC warns on any usage of ``long double`` in device code, because ``long double`` will be demoted to ``double`` in device code.
  This warning can be suppressed silenced with ``#pragma``\ s, but only globally, not just when using ``complex``.
  User-defined floating-point literals must be specified in terms of ``long double``, so they lead to warnings
  that are unable to be suppressed.

Extensions
--------------

- Handling of infinities

  Our implementation by default recovers infinite values during multiplication and division. This adds a significant runtime overhead,
  so we allow disabling that canonicalization if it is not desired.

  Definition of ``LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS`` disables canonicalization for both multiplication *and* division.

  Definition of ``LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_MULTIPLICATION`` or ``LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_DIVISION`` disables
  canonicalization for multiplication or division individually.

- Support for half and bfloat16 (since libhipcxx 2.4.0)

  Our implementation includes support for the ``__half`` type from ``<cuda_fp16.h>``, when the CUDA toolkit version is at
  least 12.2, and when ``CCCL_DISABLE_FP16_SUPPORT`` is **not** defined.

  This is detected automatically when compiling through NVCC. If you are compiling a host-only translation unit directly
  with the host compiler, you must define the macro ``LIBCUDACXX_ENABLE_HOST_NVFP16`` prior to including any libhipcxx headers,
  and you must ensure that the ``<cuda_fp16.h>`` header that's found by the compiler comes from a CUDA toolkit version
  12.2 or higher.

  Our implementation includes support for the ``__nv_bfloat16`` type from ``<cuda_bf16.h>``, when the conditions for the
  support of ``__half`` are fulfilled, and when ``CCCL_DISABLE_BF16_SUPPORT`` and ``CCCL_DISABLE_FP16_SUPPORT`` are **not** defined.

- C++20 constexpr ``<complex>`` is available in C++14.
