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
  :description: API reference for cuda::stream_ref and cuda::stream, the non-owning and owning wrappers around a HIP stream in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, stream, stream_ref, hipStream_t, asynchronous execution

.. _cccl-runtime-stream:

Streams
========

This page documents ``cuda::stream_ref`` and ``cuda::stream``, which wrap a ``hipStream_t``.

Stream is conceptually a queue of operations for a specific device. It is passed as an argument to all asynchronous operations like kernel launch, memory copy and allocations.
See `Streams and concurrent execution <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html#streams-and-concurrent-execution>`__ for an introduction to HIP streams.

:cpp:class:`cuda::stream_ref`
-------------------------------
.. _cccl-runtime-stream-stream-ref:

:cpp:class:`cuda::stream_ref` is a non-owning wrapper around a ``hipStream_t``. It prevents unsafe implicit constructions from
``nullptr`` or integer literals and provides convenient helpers for:

- ``sync()``: wait for the recorded work to complete
- ``is_done()``: non-blocking completion query
- comparison operators against other :cpp:class:`cuda::stream_ref` or ``hipStream_t``

Availability: libhipcxx 2.7

Example:

.. code:: cpp

    #include <cuda/stream>

    hipStream_t stream;
    hipStreamCreate(&stream);
    cuda::stream_ref ref{stream};

    ref.sync();            // synchronizes the stream via hipStreamSynchronize
    assert(ref.is_done()); // verifies that the stream has finished all operations via hipStreamQuery

    // compare against other stream_ref or hipStream_t
    assert(ref == stream);
    assert(ref != cuda::invalid_stream);

    hipStreamDestroy(stream);

:cpp:struct:`cuda::stream`
---------------------------
.. _cccl-runtime-stream-stream:

:cpp:struct:`cuda::stream` is an owning wrapper around a ``hipStream_t`` that manages the lifetime of the underlying HIP
stream.
It derives from :cpp:class:`cuda::stream_ref`, provides all of its functionality, and can be used anywhere a
:cpp:class:`cuda::stream_ref` is expected.
It can be constructed for a specific :cpp:class:`cuda::device_ref`, moved (but not copied), and converted from or to a
``hipStream_t`` via ``from_native_handle``/``release()``.

Availability: libhipcxx 3.4

.. code:: cpp

   #include <cuda/stream>
   #include <cuda/devices>

   int main() {
     {
       // Create a stream on a specific device
       cuda::stream s{cuda::devices[0]};

      // Pass to a stream-ordered API

       // Synchronize the stream
       s.sync();
     } // Stream is automatically destroyed here
   }
