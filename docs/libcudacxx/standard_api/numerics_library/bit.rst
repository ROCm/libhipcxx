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
  :description: Documents cuda::std::bit in libhipcxx, covering bit manipulation functions such as popcount, rotl, rotr, bit_width, and their SASS instruction mappings for GPU performance.
  :keywords: libhipcxx, ROCm, HIP, C++, bit manipulation, popcount, bitwise, SASS, nodiscard

.. _libcudacxx-standard-api-numerics-bit:

``<cuda/std/bit>``
==================

This page documents ``cuda::std::bit`` in libhipcxx, covering bit manipulation functions such as popcount, rotl, rotr, and bit_width, with notes on GPU SASS instruction mappings.

CUDA Performance Considerations
-------------------------------

Given an unsigned integer with ``N`` bits and ``N <= 32``, the ``<bit>`` functions translate into the following SASS instructions. For some functions, the result is decorated with a compile-time assumption to restrict its range and allow further optimizations.

- ``bit_width()`` translates into a single ``FLO`` SASS instruction. The result is assumed to be in the range ``[0, N]``.
- ``bit_ceil()`` translates into ``ADD, FLO, SHL, IMINMAX`` SASS instructions. The result is assumed to be greater than or equal to the input.
- ``bit_floor()`` translates into ``FLO, SHL`` SASS instructions. The result is assumed to be less than or equal to the input.
- ``byteswap()`` translates into a single ``PRMT`` SASS instruction.
- ``popcount()`` translates into a single ``POPC`` SASS instruction. The result is assumed to be in the range ``[0, N]``.
- ``has_single_bit()`` translates into ``POPC + ISETP`` SASS instructions.
- ``rotl()/rotr()`` translate into a single ``SHF`` (funnel shift) SASS instruction.
- ``countl_zero()`` translates into ``FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N]``.
- ``countl_one()`` translates into ``LOP3, FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N]``.
- ``countr_zero()`` translates into ``BREV, FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N]``.
- ``countr_one()`` translates into ``LOP3, BREV, FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N]``.

Additional Notes
----------------

- All functions are marked ``[[nodiscard]]`` and ``noexcept``.
- All functions support 128-bit integer types.
- ``bit_ceil()`` checks for overflow in debug mode.
- ``rotl()/rotr()`` checks for invalid count value (``INT_MIN``) in debug mode.

.. note::

    When the input values are run-time values that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations, using the functions could not be optimal.

.. note::

    GCC <= 8 uses a slow path with more instructions even in CUDA.
