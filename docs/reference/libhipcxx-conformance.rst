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
  :description: libhipcxx conformance and ABI evolution
  :keywords: libhipcxx, ROCm, HIP, conformance, ABI, versioning, C++ standard

.. _libhipcxx-conformance:

********************************************************************
Conformance and ABI evolution
********************************************************************

Conformance
===========

libhipcxx aims to be a conforming implementation of the C++ Standard,
`ISO/IEC IS 14882 <https://eel.is/c++draft>`_, Clause 16 through 32.

ABI evolution
=============

libhipcxx does not maintain long-term ABI stability. Promising long-term ABI stability would prevent
fixing mistakes and providing best in class performance, so no such promises are made.

The ABI is broken at every major release. The life cycle of an ABI version is approximately one
year, and long-term support for an ABI version ends after approximately two years.

The latest ABI version is always the default. For the ABI version associated with each release, see
the :ref:`releases <libcudacxx-releases>` section.
