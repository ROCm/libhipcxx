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
  :description: API reference for cuda::fast_mod_div, which precomputes a divisor for fast integer division and modulo in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, fast_mod_div, integer division, modulo, div, math

.. _libcudacxx-extended-api-math-fast-mod-div:

``cuda::fast_mod_div``
======================

This page documents ``cuda::fast_mod_div``, which precomputes a divisor to speed up subsequent integer division and modulo operations.

Defined in the ``<cuda/cmath>`` header.

.. code:: cpp

    namespace cuda {

    template <typename T, bool DivisorIsNeverOne = false>
    class fast_mod_div {
    public:
        fast_mod_div() = delete;

        __host__ __device__
        explicit fast_mod_div(T divisor) noexcept;

        template <typename U>
        [[nodiscard]] __host__ __device__ friend
        cuda:::std::common_type_t<T, U> operator/(U dividend, fast_mod_div<T> divisor) noexcept;

        template <typename U>
        [[nodiscard]] __host__ __device__ friend
        cuda:::std::common_type_t<T, U> operator%(U dividend, fast_mod_div<T> divisor) noexcept;

        [[nodiscard]] __host__ __device__
        operator T() const noexcept;
    };

    } // namespace cuda

.. code:: cpp

    namespace cuda {

    template <typename T, typename U>
    [[nodiscard]] __host__ __device__
    cuda::std::pair<T, U> div(T dividend, fast_mod_div<U> divisor) noexcept;

    } // namespace cuda

The class ``fast_mod_div`` is used to pre-compute the modulo and division of an integer value, to be used in a second stage for efficiency: :math:`floor\left(\frac{dividend}{divisor}\right)`.

**Parameters**

- ``divisor``:  The divisor.
- ``dividend``: The dividend.
- ``DivisorIsNeverOne``: Indicates that ``divisor != 1`` and skips one comparison in the second stage.

**Constraints**

- ``T`` and ``U`` are integer types.
- ``max_value(dividend type) <= max_value(divisor type)``.

**Preconditions**

- ``divisor > 0``.
- ``dividend >= 0``.
- ``divisor > 1`` if ``DivisorIsNeverOne == true``.

**Performance considerations**

- ``fast_mod_div`` needs to be initialized on the host and executed on the device for optimal performance.
- ``T`` signed type ensures the best performance.
- Larger types (> 32-bits) are slower than smaller types.
- ``DivisorIsNeverOne == true`` can be used to skip one comparison.
- ``__builtin_assume(dividend != cuda::std::numeric_limits<U>::max())`` can be used to skip one comparison with unsigned values.

..
   ``T == int`` translates to ``SEL``, ``IMAD``, and x2 ``SHF`` instructions.

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void div_kernel(cuda::fast_mod_div<int> divisor) {
        assert(45 / divisor == 2);
        assert(45 % divisor == 5);
        assert((cuda::div(45, divisor) == cuda::std::pair{2, 5}));
    }

    int main() {
        cuda::fast_mod_div<int> divisor(20);
        div_kernel<<<1, 1>>>(divisor);
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/fM7E9v9aP>`__
