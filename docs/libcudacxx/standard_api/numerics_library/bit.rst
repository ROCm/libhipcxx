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
  :description: Documents cuda::std::bit in libhipcxx, covering bit manipulation functions such as popcount, rotl, rotr, and bit_width in host and device code.
  :keywords: libhipcxx, ROCm, HIP, C++, bit manipulation, popcount, bitwise, nodiscard

.. _libcudacxx-standard-api-numerics-bit:

``<cuda/std/bit>``
==================

This page documents ``<cuda/std/bit>`` in libhipcxx, which provides the C++ Standard Library bit manipulation
functions, such as ``popcount``, ``rotl``, ``rotr``, and ``bit_width``, in host and device code.

``cuda::std::bit_cast``
-----------------------

``cuda::std::bit_cast`` extended the standard ``std::bit_cast`` to also recognize the HIP extended floating-point scalar and vector types as trivially copyable.

**Limitations**

- The function can be used in ``constexpr`` contexts only when the source and destination types are trivially copyable.
- The function cannot be used in ``constexpr`` contexts with MSVC <= 19.25 and GCC <= 10.

Notes
-----

- All functions are marked ``[[nodiscard]]`` and ``noexcept``.
- All functions support 128-bit integer types.
- ``bit_ceil()`` checks for overflow in debug mode.
- ``rotl()/rotr()`` checks for invalid count value (``INT_MIN``) in debug mode.
- The functions are ``constexpr``: for compile-time inputs libhipcxx evaluates them with a portable
  implementation, so the compiler folds them. For runtime inputs in device code it dispatches to the HIP device
  intrinsics, such as ``__popc`` for ``popcount()`` and ``__clz`` for ``countl_zero()``, while ``byteswap()`` uses
  ``__builtin_bswap32``.
