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
  :description: API reference for cuda::std::memset and cuda::std::memcpy from <cuda/std/cstring>, including their debug-mode preconditions, in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, cstring, memset, memcpy, preconditions

.. _libcudacxx-standard-api-cstring:

``<cuda/std/cstring>``
======================

This page documents ``<cuda/std/cstring>``, which provides the byte-wise memory functions ``cuda::std::memset`` and
``cuda::std::memcpy`` for host and device code.

``cuda::std::memset``
---------------------

.. code:: cpp

   __host__ __device__
   inline void* memset(void* dest, int ch, size_t count) noexcept;

See `std::memset <https://en.cppreference.com/w/cpp/string/byte/memset.html>`_ for the full documentation.

**Preconditions**

The following preconditions are only enabled with libhipcxx 3.4 or later:

    - ``dest`` is a valid pointer.
    - ``dest + count`` is a valid pointer.

A valid pointer is one that is not NULL and within the correct range if it belongs to the shared memory address space.

----

``cuda::std::memcpy``
---------------------

.. code:: cpp

   __host__ __device__
   inline void* memcpy(void* dest, const void* src, size_t count) noexcept;

See `std::memcpy <https://en.cppreference.com/w/cpp/string/byte/memcpy.html>`_  for the full documentation.

**Preconditions**

The following preconditions are only enabled with libhipcxx 3.4 or later:

    - ``src`` is a valid pointer.
    - ``src + count`` is a valid pointer.
    - ``dest`` is a valid pointer.
    - ``dest + count`` is a valid pointer.
    - ``src`` and ``dest`` don't overlap.

A valid pointer is one that is not NULL and within the correct range if it belongs to the shared memory address space.
