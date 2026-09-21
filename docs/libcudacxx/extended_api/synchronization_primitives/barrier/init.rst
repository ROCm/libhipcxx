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

.. _libcudacxx-extended-api-synchronization-barrier-barrier-init:

cuda::barrier::init
=======================

Defined in header ``<cuda/barrier>``:

.. code:: cuda

   template <cuda::thread_scope Scope,
             typename CompletionFunction = /* unspecified */>
   class barrier {
   public:
     // ...

     __host__ __device__
     friend void init(cuda::std::barrier* bar,
                      cuda::std::ptrdiff_t expected,
                      CompletionFunction cf = CompletionFunction{});
   };

The friend function ``cuda::barrier::init`` may be used to initialize an
:ref:`cuda::barrier <libcudacxx-extended-api-synchronization-barrier>` that has not been initialized.

When using libhipcxx with NVCC, ``__shared__`` ``cuda::barrier`` will not have its constructors run because ``__shared__``
variables are not initialized. ``cuda::barrier::init`` should be used to properly initialize such a
:ref:`cuda::barrier <libcudacxx-extended-api-synchronization-barrier>`.

An NVCC diagnostic warning about the ignored constructor will be emitted:

.. code:: bash

   warning: dynamic initialization is not supported for a function-scope static
   __shared__ variable within a __device__/__global__ function

It can be silenced using ``#pragma nv_diag_suppress static_var_with_dynamic_init``.

Example
-------

.. code:: cuda

   #include <cuda/barrier>

   // Disables `cuda::barrier` initialization warning.
   #pragma nv_diag_suppress static_var_with_dynamic_init

   __global__ void example_kernel() {
     __shared__ cuda::barrier<cuda::thread_scope_block> bar;
     init(&bar, 1);
   }

`See it on Godbolt <https://godbolt.org/z/nK5q3xh34>`_
