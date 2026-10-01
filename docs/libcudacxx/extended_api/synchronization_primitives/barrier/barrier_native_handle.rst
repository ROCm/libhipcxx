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
  :description: API reference for cuda::device::barrier_native_handle, which returns a pointer to the native PTX mbarrier handle of a block-scoped shared memory barrier in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, barrier_native_handle, barrier, PTX, mbarrier, shared memory, thread_scope_block

.. _libcudacxx-extended-api-synchronization-barrier-barrier-native-handle:

``cuda::device::barrier_native_handle``
=======================================

This page documents ``cuda::device::barrier_native_handle``, which returns a pointer to the native PTX mbarrier handle of a block-scoped shared memory barrier.

Defined in header ``<cuda/barrier>``:

.. code:: cuda

   __device__ cuda::std::uint64_t* cuda::device::barrier_native_handle(
     cuda::barrier<cuda::thread_scope_block>& bar);

Returns a pointer to the native handle of a :ref:`cuda::barrier <libcudacxx-extended-api-synchronization-barrier>`
if its scope is ``cuda::thread_scope_block`` and it is allocated in shared memory.
The pointer is suitable for use with PTX instructions.

Notes
-----

If ``bar`` is not in ``__shared__`` memory, the behavior is undefined.

Return Value
------------

A pointer to the PTX "mbarrier" subobject of the :ref:`cuda::barrier <libcudacxx-extended-api-synchronization-barrier>`
object.

Example
-------

.. code:: cuda

   #include <cuda/barrier>

   __global__ void example_kernel(cuda::barrier<cuda::thread_scope_block>& bar) {
     auto ptr = cuda::device::barrier_native_handle(bar);

     asm volatile (
         "mbarrier.arrive.b64 _, [%0];"
         :
         : "l" (ptr)
         : "memory");
     // Equivalent to: `(void)b.arrive()`.
   }

`See it on Godbolt <https://godbolt.org/z/dr4798Y76>`_
