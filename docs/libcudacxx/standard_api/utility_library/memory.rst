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
  :description: Documents cuda::std::memory in libhipcxx, providing addressof, align, assume_aligned, and uninitialized memory algorithms with constexpr support from C++11 onwards.
  :keywords: libhipcxx, ROCm, HIP, C++, memory, addressof, align, assume_aligned, uninitialized

.. _libcudacxx-standard-api-utility-memory:

``<cuda/std/memory>``
=====================

This page documents the ``cuda::std`` memory utilities available in libhipcxx, including addressof, align, assume_aligned, and uninitialized memory algorithms.

Provided functionalities
------------------------

- ``cuda::std::addressof``. See the C++ documentation of `std::addressof <https://en.cppreference.com/w/cpp/memory/addressof>`_.
- ``cuda::std::align``. See the C++ documentation of `std::align <https://en.cppreference.com/w/cpp/memory/align>`_.
- ``cuda::std::assume_aligned``. See the C++ documentation of `std::assume_aligned <https://en.cppreference.com/w/cpp/memory/assume_aligned>`_.
- Uninitialized memory algorithms. See the C++ documentation `<https://en.cppreference.com/w/cpp/memory>`_.

Extensions
----------

- Most features are available from C++11 onwards.
- ``cuda::std::addressof`` is constexpr from C++11 on if compiler support is available.
- ``cuda::std::assume_aligned`` is constexpr from C++14 on.

Restrictions
------------

- `construct_at` is only available in C++20 as that is explicitly mentioned in the standard.
- The specialized memory algorithms are not parallel.
