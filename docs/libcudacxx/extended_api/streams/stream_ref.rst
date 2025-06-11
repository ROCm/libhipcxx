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
  :description: API reference for cuda::stream_ref, a type-safe wrapper around hipStream_t that prevents implicit conversions and provides wait and ready member functions in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, stream_ref, hipStream_t, stream, wait, ready, implicit conversion

.. _libcudacxx-extended-api-streams-stream-ref:

``cuda::stream_ref``
====================

This page documents ``cuda::stream_ref``, a type-safe wrapper around ``hipStream_t`` that prevents implicit conversions and provides wait and ready member functions.

HIP `stream-ordered allocations <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html>`__
rely on ``hipStream_t`` as a handle to the HIP stream.

However, as this is just an alias for a plain pointer type it carries with it common pitfalls around implicit
conversions from, for example, ``nullptr`` or a literal ``0``.

These hard to spot bugs can be avoided through ``cuda::stream_ref``, which is a simple wrapper around a ``hipStream_t``
that prevents implicit conversions. It also provides the ``wait()`` and ``ready()`` member functions to facilitate
waiting for a stream to finish and checking whether it is finished.

.. code:: cpp

       hipStream_t stream;
       hipStreamCreate(&stream);
       cuda::stream_ref ref{stream};

       ref.wait();          // synchronizes the stream via hipStreamSynchronize
       assert(ref.ready()); // verifies that the stream has finished all operations via hipStreamQuery
       hipStreamDestroy(stream);
