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
  :description: Documents cuda::std::numeric in libhipcxx, providing constexpr numeric algorithms available from C++11 onwards, with notes on omitted parallel algorithms.
  :keywords: libhipcxx, ROCm, HIP, C++, numeric algorithms, constexpr, nodiscard, parallel algorithms

.. _libcudacxx-standard-api-numerics-numeric:

``<cuda/std/numeric>``
======================

This page documents ``cuda::std::numeric`` in libhipcxx, which provides constexpr numeric algorithms including iota, accumulate, and inner_product, available from C++11 onwards.

Omissions
---------

-  Currently we do not expose any parallel algorithms.

Extensions
----------

- All features of ``<numeric>`` are made available in C++11 onwards.
- All features of ``<numeric>`` are made constexpr in C++14 onwards.
- Algorithms that return a value and not an iterator have been marked ``[[nodiscard]]``.


..
   Parallel algorithms are not yet supported in libhipcxx.

   Parallel standard algorithms
   ----------------------------

   CCCL provides an implementation for the standard `parallel algorithms library <http://www.eel.is/c++draft/algorithms.parallel>`_

   Currently the CUDA backend is the only supported backend. It can be selected by passing the `cuda::execution::gpu`
   execution policy to one of the supported algorithms. The CUDA backend requires the passed in sequences to reside in
   device accessible memory and the iterators into those sequences to be at least random access iterators. The CUDA backend
   is enabled if the program is compiled with a CUDA compiler in CUDA mode.

   The use of any other execution policy is currently not supported and results in a compile time error.

   The following algorithms are supported:

     * ``adjacent_difference``
     * ``exclusive_scan``
     * ``inclusive_scan``
     * ``transform_exclusive_scan``
     * ``transform_inclusive_scan``
     * ``reduce``
     * ``transform_reduce``
