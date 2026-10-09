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
  :description: API reference for cuda::static_for, which provides a for loop with compile-time indices in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, static_for, compile-time loop, loop unrolling, integral_constant, utility

.. _libcudacxx-extended-api-utility-static-for:

``static_for``
==============

This page documents ``cuda::static_for``, which executes a callable for each index of a compile-time range.

Defined in ``<cuda/utility>`` header.

.. code:: cpp

    namespace cuda {

    template <auto Size, typename Operator, typename... TArgs>
    __host__ __device__ constexpr
    void static_for(Operator op, TArgs&&... args) noexcept(/*see-below*/); // (1)

    template <auto Start, decltype(Start) End, decltype(Start) Step = 1, typename Operator, typename... TArgs>
    __host__ __device__ constexpr
    void static_for(Operator op, TArgs&&... args) noexcept(/*see-below*/); // (2)

    template <typename T, T Size, typename Operator, typename... TArgs>
    __host__ __device__ constexpr
    void static_for(Operator op, TArgs&&... args) noexcept(/*see-below*/); // (3)

    template <typename T, T Start, T End, T Step = 1, typename Operator, typename... TArgs>
    __host__ __device__ constexpr
    void static_for(Operator op, TArgs&&... args) noexcept(/*see-below*/); // (4)

    } // namespace cuda

| The functionality provides a ``for`` loop with compile-time indices.
| ``static_for`` is available in two forms:

- Executes ``op`` for each value in the range ``[0, Size)`` (1, 3).
- Executes ``op`` for each value in the range ``[Start, End)`` with step ``Step`` (2, 4).

| The function is ``noexcept`` if all invocations of ``op`` with ``integral_constant</*index-type*/, /*index-value*/>`` and the ``args...`` are *non-throwing*. Only visited indices participate in the ``noexcept`` evaluation.

**Parameters**

- ``Size``: the number of iterations (1, 3).
- ``Start``, ``End``, ``Step``: the start, end, and step of the range. Note that ``End`` and ``Step`` are converted to the type of ``Start`` (2, 4).
- ``T``: type of the loop index (3, 4).
- ``op``: the function to execute.
- ``args``: additional arguments to pass to ``op``.

``op`` is a callable object that accepts an ``integral_constant`` of the same type of ``Size`` or ``Start``.

**Performance considerations**

- The functions are useful as metaprogramming utility and when a loop requires full unrolling, independently of the compiler's constrains, optimization level, and heuristics. In addition, the index is a compile-time constant, which can be used in a constant expression and further optimize the code.

- Conversely, ``static_for`` is more expensive to compile compared to ``#pragma unroll``. Additionally, the preprocessor directive interacts with the compiler, which tunes the loop unrolling based on register usage, binary size, and instruction cache.

Example
-------

.. code:: cpp

    #include <cuda/utility>
    #include <cstdio>

    __global__ void kernel() {
        cuda::static_for<5>([](auto i){ static_assert(i >= 0 && i < 5); });

        cuda::static_for<5>([](auto i){ printf("%d, ", i()); }); // 0, 1, 2, 3, 4,
        printf("\n");

        cuda::static_for<short, 5>([](auto i){ printf("%d, ", i()); }); // 0, 1, 2, 3, 4,
        printf("\n");

        cuda::static_for<-5, 7, 3>([](auto i){ printf("%d, ", i()); }); // -5, -2, 1, 4,
        printf("\n");

        cuda::static_for<5>([](auto i){
            if constexpr (i > 0) {
                cuda::static_for<i()>([](auto j){ printf("%d, ", j()); });
                printf("\n");
            }
        });
        // 0,
        // 0, 1,
        // 0, 1, 2,
        // 0, 1, 2, 3,
        cuda::static_for<5>([](auto i, int a, int b, int c){}, 1, 2, 3); // 1, 2, 3 optional arguments
    }

    int main() {
        kernel<<<1, 1>>>();
        hipDeviceSynchronize();
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/1GWc4dqKj>`_
