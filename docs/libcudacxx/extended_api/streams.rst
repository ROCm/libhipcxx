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
  :description: Overview of the libhipcxx streams extended API, including the stream_ref wrapper around hipStream_t for type-safe stream management in HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, streams, stream_ref, hipStream_t, stream-ordered, memory allocation

.. _libcudacxx-extended-api-streams:

Streams
=======

This page covers the streams extended API, providing ``cuda::stream_ref`` as a type-safe wrapper around ``hipStream_t`` that prevents common implicit-conversion pitfalls.

.. toctree::
   :hidden:
   :maxdepth: 1

   cuda::stream_ref <streams/stream_ref>

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - API
     - Description
     - Since

   * - :ref:`stream_ref <libcudacxx-extended-api-streams-stream-ref>`
     - A wrapper around a ``hipStream_t``
     - libhipcxx 2.7
