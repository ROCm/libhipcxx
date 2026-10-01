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
  :description: API reference for cuda::stream_ref, a type-safe wrapper around cudaStream_t that prevents implicit conversions and provides wait and ready member functions in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, stream_ref, cudaStream_t, stream, wait, ready, implicit conversion

.. _libcudacxx-extended-api-streams-stream-ref:

``cuda::stream_ref``: a wrapper around a ``cudaStream_t``
==========================================================

This page documents ``cuda::stream_ref``, a type-safe wrapper around ``cudaStream_t`` that prevents implicit conversions and provides wait and ready member functions.

CUDA `stream-ordered allocations <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator>`__
rely on ``cudaStream_t`` as a handle to the cuda stream.

However, as this is just an alias for a plain pointer type it carries with it common pitfalls around implicit
conversions from, for example, ``nullptr`` or a literal ``0``.

These hard to spot bugs can be avoided through ``cuda::stream_ref``, which is a simple wrapper around a ``cudaStream_t``
that prevents implicit conversions. It also provides the ``wait()`` and ``ready()`` member functions to facilitate
waiting for a stream to finish and checking whether it is finished.

.. code:: cpp

       cudaStream_t stream;
       cudaStreamCreate(&stream);
       cuda::stream_ref ref{stream};

       ref.wait();          // synchronizes the stream via cudaStreamSynchronize
       assert(ref.ready()); // verifies that the stream has finished all operations via cudaStreamQuery
       cudaStreamDestroy(stream);
