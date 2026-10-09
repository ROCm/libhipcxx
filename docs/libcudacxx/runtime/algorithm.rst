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
  :description: API reference for cuda::copy_bytes and cuda::fill_bytes, stream-ordered byte-wise copy and fill primitives in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, copy_bytes, fill_bytes, stream-ordered, span, mdspan

.. _cccl-runtime-algorithm:

Algorithm
==========

This page documents ``cuda::copy_bytes`` and ``cuda::fill_bytes``, which enqueue byte-wise copies and fills on a stream.

The ``runtime`` part of the ``cuda/algorithm`` header provides stream-ordered, byte-wise primitives that operate on
:cpp:class:`cuda::std::span` and :cpp:class:`cuda::std::mdspan`-compatible types. They require a
:cpp:class:`cuda::stream_ref` to enqueue work.

:cpp:func:`cuda::copy_bytes`
-------------------------------
.. _cccl-runtime-algorithm-copy_bytes:

Launch a byte-wise copy from source to destination on the provided stream.

- Signature: :cpp:func:`cuda::copy_bytes`
- Overloads accept :cpp:class:`cuda::std::span`-convertible contiguous ranges or
  :cpp:class:`cuda::std::mdspan`-convertible multi-dimensional views.
- Elements must be trivially copyable
- :cpp:class:`cuda::std::mdspan`-convertible types must convert to an mdspan that is exhaustive
- The optional ``config`` argument is a :cpp:struct:`cuda::copy_configuration`. In libhipcxx the copy is enqueued with
  ``hipMemcpyAsync`` and the configuration is accepted but ignored; ``cuda::source_access_order`` only provides
  the ``any`` enumerator.

..
   Not supported in libhipcxx: source access order and managed-memory location hints require a
   CUDA 13 memcpy-with-attributes API that HIP does not provide.

   - The optional ``config`` argument is a :cpp:struct:`cuda::copy_configuration` that controls source access order and
     managed-memory location hints

Availability: libhipcxx 3.4

.. code:: cpp

   #include <cuda/algorithm>
   #include <cuda/stream>
   #include <cuda/std/algorithm>
   #include <cuda/std/span>

   void copy_example(cuda::stream_ref s, cuda::std::span<const int> src, cuda::std::span<int> dst) {
     // copy_bytes copies up to src.size_bytes(); dst can be larger.
     auto n = cuda::std::min(src.size(), dst.size());
     auto src_prefix = src.first(n);
     auto dst_prefix = dst.first(n);

     // Enqueue a stream-ordered byte-wise copy on stream s.
     cuda::copy_bytes(s, src_prefix, dst_prefix);
   }

..
   Not supported in libhipcxx: cuda::source_access_order::during_api_call is not available on HIP.

   .. code:: cpp

      // Advanced behavior: customize source access order for this copy.
      auto config = cuda::copy_configuration{
        .src_access_order = cuda::source_access_order::during_api_call,
      };
      cuda::copy_bytes(s, src_suffix, dst_suffix, config);


:cpp:func:`cuda::fill_bytes`
-------------------------------
.. _cccl-runtime-algorithm-fill_bytes:

Launch a byte-wise fill of the destination on the provided stream.

- Overloads accept :cpp:class:`cuda::std::span`-convertible or :cpp:class:`cuda::std::mdspan`-convertible destinations.
- Elements must be trivially copyable
- :cpp:class:`cuda::std::mdspan`-convertible types must convert to an mdspan that is exhaustive

Availability: libhipcxx 3.4

.. code:: cpp

   #include <cuda/algorithm>
   #include <cuda/stream>
   #include <cuda/std/algorithm>
   #include <cuda/std/span>

   void fill_example(cuda::stream_ref s, cuda::std::span<unsigned char> dst) {
     // Reserve 16-byte red zones at both ends and clear the payload in between.
     auto guard = cuda::std::min(static_cast<decltype(dst.size())>(16), dst.size() / 2);
     auto head  = dst.first(guard);
     auto body  = dst.subspan(guard, dst.size() - 2 * guard);
     auto tail  = dst.last(guard);

     cuda::fill_bytes(s, head, 0xCD); // debug guard pattern
     cuda::fill_bytes(s, body, 0x00); // initialize payload
     cuda::fill_bytes(s, tail, 0xCD); // debug guard pattern
   }
