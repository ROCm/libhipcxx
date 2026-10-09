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
  :description: API reference for cuda::saturating_overflow_cast, which performs a saturating integer cast with overflow checking in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, saturating_overflow_cast, saturation, cast, overflow, numeric

.. _libcudacxx-extended-api-numeric-saturating_overflow_cast:

``cuda::saturating_overflow_cast``
=====================================

This page documents ``cuda::saturating_overflow_cast``, which casts an integer value to another integer type with saturation and overflow checking.

.. code:: cpp

   template <class T>
   struct overflow_result;

   template <class To, class From>
   [[nodiscard]] __host__ __device__ inline constexpr
   overflow_result<To> saturating_overflow_cast(From from) noexcept;

The function ``cuda::saturating_overflow_cast`` does saturating cast of a value of type ``From`` to type ``To`` with overflow checking.

**Parameters**

- ``from``: The value to be casted.

**Return value**

- Returns an :ref:`overflow_result <libcudacxx-extended-api-numeric-overflow_result>` object that contains the result of the saturating cast and a boolean indicating whether an overflow occurred.

**Constraints**

- ``To`` and ``From`` must be `integer types <https://eel.is/c++draft/basic.fundamental#1>`_.

Example
-------

.. code:: cpp

    #include <cuda/numeric>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>

    __global__ void kernel()
    {
        constexpr auto int_max = cuda::std::numeric_limits<int>::max();
        constexpr auto int_min = cuda::std::numeric_limits<int>::min();

        if (auto result = cuda::saturating_overflow_cast<unsigned>(int_max))
        {
            assert(false); // Should not be reached
        }
        else
        {
            assert(result.value == static_cast<unsigned>(int_max));
        }

        if (auto result = cuda::saturating_overflow_cast<unsigned>(int_min)) // saturated
        {
            assert(result.value == 0);
        }
        else
        {
            assert(false); // Should not be reached
        }
    }

    int main()
    {
        kernel<<<1, 1>>>();
        hipDeviceSynchronize();
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/oKW81Eajx>`_
