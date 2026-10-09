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
  :description: API reference for cuda::always_true and cuda::always_false, function objects that always return true or false in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, always_true, always_false, function objects, functional, constexpr

.. _libcudacxx-extended-api-functional-always-true-false:

``cuda::always_true`` and ``cuda::always_false``
================================================

This page documents ``cuda::always_true`` and ``cuda::always_false``, function objects that always return ``true`` or ``false``.

Defined in the header ``<cuda/functional>``.

.. code:: cpp

    struct always_true {
        template <typename... Ts>
        [[nodiscard]] __host__ __device__ constexpr bool operator()(Ts&&...) const noexcept;
    };

    struct always_false {
        template <typename... Ts>
        [[nodiscard]] __host__ __device__ constexpr bool operator()(Ts&&...) const noexcept;
    };

``cuda::always_true`` is a function object that always returns ``true`` regardless of the number and type of arguments
passed. ``cuda::always_false`` is a function object that always returns ``false`` regardless of the number and type of
arguments passed.

Both types are empty, trivially copyable, and their ``operator()`` is ``constexpr`` and ``noexcept``.

Example
-------

.. code:: cpp

    #include <cuda/functional>

    __global__ void example_kernel() {
        cuda::always_true  pred_true{};
        cuda::always_false pred_false{};

        // Returns true regardless of arguments
        static_assert(pred_true());
        static_assert(pred_true(1, 2, 3));

        // Returns false regardless of arguments
        static_assert(!pred_false());
        static_assert(!pred_false(1, 2, 3));
    }
