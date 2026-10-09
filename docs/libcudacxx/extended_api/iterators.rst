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
  :description: Overview of the fancy iterators provided by libhipcxx for HIP, which avoid storing data in memory and fuse multiple iterations.
  :keywords: libhipcxx, ROCm, HIP, C++, iterators, fancy iterators, counting_iterator, transform_iterator, zip_iterator

.. _libcudacxx-extended-api-iterators:

Fancy Iterators
---------------

libhipcxx provides a set of fancy iterators that originate from ``Thrust``.
They allow the user to avoid storing data needlessly in memory and fuse multiple iterations into a single run.

The following fancy iterators are available from ``<cuda/iterator>``:

- ``cuda::constant_iterator``
- ``cuda::counting_iterator``
- ``cuda::discard_iterator``
- ``cuda::permutation_iterator``
- ``cuda::shuffle_iterator``
- ``cuda::strided_iterator``
- ``cuda::tabulate_output_iterator``
- ``cuda::transform_input_output_iterator``
- ``cuda::transform_iterator``
- ``cuda::transform_output_iterator``
- ``cuda::zip_iterator``
- ``cuda::zip_transform_iterator``
- ``cuda::zip_function``

..
   The upstream page links the Doxygen-generated API reference (../api/class*__iterator,
   ../api/class*zip__function, ../api/group__iterators*), which the libhipcxx documentation
   build does not generate.
