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
  :description: API reference for cuda::sincos, which computes the sine and cosine of a value at the same time in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, sincos, sine, cosine, sincos_result, math

.. _libcudacxx-extended-api-math-sincos:

``cuda::sincos``
====================================

This page documents ``cuda::sincos``, which computes the sine and cosine of a value at the same time.

Defined in the ``<cuda/cmath>`` header.

.. code:: cpp

   namespace cuda {

   template <class T>
   struct sincos_result
   {
     T sin;
     T cos;
   };

   template </*floating-point-type*/ T>
   [[nodiscard]] __host__ __device__
   sincos_result<T> sincos(T value) noexcept; // (1)

   template <class Integral>
   [[nodiscard]] __host__ __device__
   sincos_result<double> sincos(Integral value) noexcept; // (2)

   } // namespace cuda

Computes :math:`\sin value` and :math:`\cos value` at the same time using more efficient algorithms than if operations were computed separately.

**Parameters**

- ``value``: The input value.

**Return value**

- ``cuda::sincos_result`` object with both values set to ``NaN`` if the input value is :math:`\pm\infty` or ``NaN`` and to results of :math:`\sin value` and :math:`\cos value` otherwise. (1)
- if ``T`` is an integral type, the input value is treated as ``double``. (2)

**Constraints**

- ``T`` is an arithmetic type.

**Performance considerations**

- If available, the functionality is implemented by compiler builtins, otherwise fallbacks to ``cuda::std::sin(value)`` and ``cuda::std::cos(value)``.

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void sincos_kernel() {
        auto [sin_pi, cos_pi] = cuda::sincos(0.f);
        assert(sin_pi == 0.f);
        assert(cos_pi == 1.f);
    }

    int main() {
        sincos_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/99PP9s1z6>`__
