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
  :description: Documents cuda::std::utility in libhipcxx, including pair, swap, and other utility components available from C++11 with notes on missing spaceship operator support.
  :keywords: libhipcxx, ROCm, HIP, C++, utility, pair, swap, trivially_copyable, spaceship operator

.. _libcudacxx-standard-api-utility-utility:

``<cuda/std/utility>``
======================

This page documents the ``cuda::std`` utility facilities available in libhipcxx, including pair, integer sequences, swap, and forward, with notes on extensions and omissions.

See the documentation of the standard header `\<utility\> <https://en.cppreference.com/w/cpp/header/utility>`_

Extensions
----------

``pair`` has been made ``trivially_copyable`` in 2.3.0.

Omissions
---------

Prior to version 2.3.0 only ``pair`` is available.

Since 2.3.0 we have implemented almost all functionality of
``<utility>``. Notably support for operator spaceship is missing due to
the specification relying on ``std`` types that are not accessible on
device.
