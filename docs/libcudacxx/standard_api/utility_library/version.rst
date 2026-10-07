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
  :description: Documents cuda::std::version in libhipcxx, providing compile-time version macros and C++ feature test macros for the libhipcxx API and ABI versions.
  :keywords: libhipcxx, ROCm, HIP, C++, version, feature test macros, ABI, API

.. _libcudacxx-standard-api-utility-version:

``<cuda/std/version>``
======================

This page documents ``cuda::std::version`` in libhipcxx, which defines version macros for tracking library and feature availability.

See the documentation of the standard header `\<version\> <https://en.cppreference.com/w/cpp/header/version>`_

Extensions
----------

The following version macros, which are explained in the :ref:`versioning section <libhipcxx-conformance>`,
are defined in this header:

- ``_LIBCUDACXX_CUDA_API_VERSION``
- ``_LIBCUDACXX_CUDA_API_VERSION_MAJOR``
- ``_LIBCUDACXX_CUDA_API_VERSION_MINOR``
- ``_LIBCUDACXX_CUDA_API_VERSION_PATCH``
- ``_LIBCUDACXX_CUDA_ABI_VERSION``
- ``_LIBCUDACXX_CUDA_ABI_VERSION_LATEST``

Restrictions
------------

``<cuda/std/version>`` includes the host standard library's ``<version>`` header, so the standard C++ feature test
macros are defined by the host standard library, not by libhipcxx. libhipcxx defines its own ``__cccl_lib_*`` macros
for the features it provides.
