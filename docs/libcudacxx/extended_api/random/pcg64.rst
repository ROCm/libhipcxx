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
  :description: API reference for cuda::pcg64, a 128-bit state PCG random number engine with 64-bit output in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, pcg64, PCG, random number engine, UniformRandomBitGenerator, random

.. _libcudacxx-extended-api-random-pcg64:

``pcg64``
=========

This page documents ``cuda::pcg64``, a PCG random number engine with 128-bit state and 64-bit output.

Defined in the ``<cuda/random>`` header.

``cuda::pcg64`` is a 128-bit state PCG XSL RR 128/64 engine that produces 64-bit unsigned integer outputs. It has a
period of ``2^128`` and supports logarithmic-time ``discard``. ``cuda::pcg64`` models the
`UniformRandomBitGenerator <https://en.cppreference.com/w/cpp/named_req/UniformRandomBitGenerator>`_ named requirement.

Example
-------

.. code:: cpp

    #include <cuda/random>

    __global__ void sample_kernel() {
        cuda::pcg64 rng(42);
        auto value = rng();
        rng.discard(10);
    }
