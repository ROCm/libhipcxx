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
  :description: Documents the ranges library support in libhipcxx, covering iterator and range concepts available from C++17 with restrictions on subsumption and omissions for range algorithms and views.
  :keywords: libhipcxx, ROCm, HIP, C++, ranges, iterator, subrange, forward_range, contiguous_range

.. _libcudacxx-standard-api-ranges:

Ranges Library
==============

This page documents the ranges and iterator library headers available in libhipcxx, covering forward, bidirectional, random-access, and contiguous range concepts.

See the documentation of the standard headers `\<iterator\> <https://en.cppreference.com/w/cpp/header/iterator>`_ and
`\<ranges\> <https://en.cppreference.com/w/cpp/header/ranges>`_

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - `\<cuda/std/iterator\> <https://en.cppreference.com/w/cpp/header/iterator>`_
     - Iterator related concepts and machinery such as ``cuda::std::forward_iterator``
     - libhipcxx 2.7
   * - `\<cuda/std/ranges\> <https://en.cppreference.com/w/cpp/header/ranges>`_
     - Range related concepts and machinery such as ``cuda::std::ranges::forward_range`` and ``cuda::std::ranges::subrange``
     - libhipcxx 2.7

Extensions
----------

All library features are available from C++17 onwards. The concepts can be used like type traits prior to C++20.

.. code:: cpp

   template<cuda::std::contiguos_range Range>
   void do_something_with_ranges_in_cpp20(Range&& range) {...}

   template<class Range, cuda::std::enable_if_t<cuda::std::contiguos_range<Range>, int> = 0>
   void do_something_with_ranges_in_cpp17(Range&& range) {...}

Restrictions
------------

- Subsumption does not work prior to C++20.

Omissions
---------

- Range-based algorithms have *not* been implemented.
- Views have *not* been implemented.
