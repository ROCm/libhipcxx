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
  :description: Learn about libhipcxx conformance with the C++ Standard and its API and ABI versioning scheme, including the version macros from <cuda/std/version>.
  :keywords: libhipcxx, ROCm, HIP, conformance, ABI, versioning, C++ standard, ABI stability, ISO

.. _libhipcxx-conformance:

********************************************************************
Conformance and versioning
********************************************************************

Conformance
===========

libhipcxx aims to be a conforming implementation of the C++ Standard,
`ISO/IEC IS 14882 <https://eel.is/c++draft>`_, Clause 16 through 32.

Versioning
==========

libhipcxx is a fork of libcu++ and follows its versioning. A libhipcxx release carries the version of the upstream
CCCL release it is based on; the current version is 3.4.0.

API version
-----------

The API version is a three-component semantic version, ``MAJOR.MINOR.PATCH``, exposed by four macros from
``<cuda/std/version>``:

- ``_LIBCUDACXX_CUDA_API_VERSION_MAJOR`` is incremented for API-breaking changes.
- ``_LIBCUDACXX_CUDA_API_VERSION_MINOR`` is incremented when API-compatible features are added.
- ``_LIBCUDACXX_CUDA_API_VERSION_PATCH`` is incremented for all other changes.
- ``_LIBCUDACXX_CUDA_API_VERSION`` combines the three as ``MAJOR * 1000000 + MINOR * 1000 + PATCH``, which is
  ``3004000`` for version 3.4.0.

A single API version is supported at a time.

ABI version
-----------

The ABI version is a single integer. libhipcxx supports exactly one ABI version, 4, so both
``_LIBCUDACXX_CUDA_ABI_VERSION`` and ``_LIBCUDACXX_CUDA_ABI_VERSION_LATEST`` from ``<cuda/std/version>`` are 4.
Defining ``_LIBCUDACXX_CUDA_ABI_VERSION`` to any other value is a compile-time error.

libhipcxx makes no promise of long-term ABI stability. Promising long-term ABI stability would prevent fixing
mistakes and achieving high performance, so no such promises are made. A program is ill-formed, no diagnostic
required, if it links translation units that were compiled against different libhipcxx versions.
