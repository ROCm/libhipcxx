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
  :description: API reference for cuda::overflow_result, which holds the result and overflow flag of arithmetic operations in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, overflow_result, overflow, integer arithmetic, numeric

.. _libcudacxx-extended-api-numeric-overflow_result:

``cuda::overflow_result``
=========================

This page documents ``cuda::overflow_result``, which holds the result of an arithmetic operation together with an overflow flag.

.. code:: cpp

   template <class T>
   struct overflow_result
   {
        T    value;
        bool overflow;

        __host__ __device__
        constexpr explicit operator bool() const noexcept;
   };

The ``overflow_result`` struct is used to represent the result of arithmetic operations that may overflow. It contains the following members:

- ``value``: The result of the operation of type ``T``.
- ``overflow``: A boolean indicating whether an overflow occurred during the operation.

The ``operator bool()`` returns ``true`` if an overflow occurred, and ``false`` otherwise.
It can be used in conditional expressions to check whether an overflow occurred.

Example:

.. code:: cpp

    auto result = /* overflow operation */;
    if (result)
    {
        // Overflow occurred
    }

**Constraints**

- ``T`` must be an integer type.
