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
  :description: API reference for cuda::maximum and cuda::minimum function objects, which compute the maximum or minimum of two values in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, maximum, minimum, function objects, functional, noexcept

.. _libcudacxx-extended-api-functional-maximum-minimum:

``cuda::maximum`` and ``cuda::minimum``
=======================================

This page documents ``cuda::maximum`` and ``cuda::minimum``, function objects that compute the maximum or minimum of two values on host and device.

Defined in the header ``<cuda/functional>``.

.. code:: cpp

    template <typename T>
    struct maximum {
        [[nodiscard]] __host__ __device__ constexpr
        T operator()(const T& a, const T& b) const noexcept(/* see below */);
    };

    template <>
    struct maximum<void> {
        template <typename T1, typename T2>
        [[nodiscard]] __host__ __device__ constexpr
        cuda::std::common_type_t<T1, T2> operator()(const T1& a, const T2& b) const noexcept(/* see below */);
    };

    template <typename T>
    struct minimum {
        [[nodiscard]] __host__ __device__ constexpr
        T operator()(const T& a, const T& b) const noexcept(/* see below */);
    };

    template <>
    struct minimum<void> {
        template <typename T1, typename T2>
        [[nodiscard]] __host__ __device__ constexpr
        cuda::std::common_type_t<T1, T2> operator()(const T1& a, const T2& b) const noexcept(/* see below */);
    };

Function objects for performing maximum and minimum operations. The ``operator()`` is ``noexcept`` when the comparison between the values is also ``noexcept``.

.. note::

   Differently from ``std::plus`` and other functional operators, ``cuda::maximum`` and ``cuda::minimum`` specialized for ``void`` returns ``cuda::std::common_type_t`` and not the implicit promotion

Floating-Point Behavior
-----------------------

For floating-point types (and extended floating-point types), ``cuda::maximum`` uses ``cuda::std::fmax`` and ``cuda::minimum`` uses ``cuda::std::fmin`` instead of the comparison operator, following the ``std::fmax``/``std::fmin`` specification for handling special values such as ``NaN``.

This also makes ``cuda::maximum`` and ``cuda::minimum`` commutative for floating-point types, unlike a plain comparison-based approach.

Example
-------

.. code:: cpp

    #include <cuda/functional>
    #include <cuda/std/cstdint>
    #include <cstdio>
    #include <numeric>

    __global__ void maximum_minimum_kernel() {
        uint16_t v1 = 7;
        uint16_t v2 = 3;
        printf("%d\n", cuda::maximum<uint16_t>{}(v1, v2)); // print "7" (uint16_t)
        printf("%d\n", cuda::minimum{}(v1, v2));           // print "3" (int)
    }

    int main() {
        maximum_minimum_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        int array[] = {3, 7, 5, 2};
        printf("%d\n", std::accumulate(array, array + 4, 0, cuda::maximum{})); // 7
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/44fdTerre>`_
