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
  :description: Documents the C++ concepts library support in libhipcxx, including fundamental library concepts available from C++14 and subsumption restrictions.
  :keywords: libhipcxx, ROCm, HIP, C++, concepts, type traits, subsumption, enable_if, cuda::std::integral

.. _libcudacxx-standard-api-concepts:

Concepts Library
================

This page documents the C++ concepts library support in libhipcxx, including fundamental library concepts available from C++14 and subsumption restrictions.

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - `\<cuda/std/concepts\> <https://en.cppreference.com/w/cpp/header/concepts>`_
     - Fundamental library concepts
     - libhipcxx 2.7

Extensions
----------

All library features are available from C++14 onwards. The concepts can be used like type traits prior to C++20.

.. code:: cpp

   template<cuda::std::integral Integer>
   void do_something_with_integers_in_cpp20(Integer&& i) {...}

   template<class Integer, cuda::std::enable_if_t<cuda::std::integral<Integer>, int> = 0>
   void do_something_with_integers_in_cpp17(Integer&& i) {...}

   template<class Integer, cuda::std::enable_if_t<cuda::std::integral<Integer>, int> = 0>
   void do_something_with_integers_in_cpp14(Integer&& i) {...}

Restrictions
------------

- Subsumption does not work prior to C++20.

  .. code:: cpp

    template<class Integer, cuda::std::enable_if_t<subsuming_concept<Integer> && true, int> = 0>
    void would_be_preferred_overload_in_cpp20(Integer&& i) {...}

    template<class Integer, cuda::std::enable_if_t<cuda::std::integral<Integer>, int> = 0>
    void is_always_ambiguous_in_cpp17(Integer&& i) {...}
