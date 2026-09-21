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

.. _libcudacxx-ptx:

PTX
=====


The ``cuda::ptx`` namespace contains functions that map one-to-one to
`PTX instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`__.
These can be used for maximal control of the generated code, or to
experiment with new hardware features before a high-level C++ API is
available.

.. toctree::
   :maxdepth: 1

   ptx/examples
   ptx/instructions

Versions and compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``cuda/ptx`` header is intended to present a stable API within one major
version of the CTK on a best effort basis. This means that:

-  All functions are marked static inline.

-  The type of a function parameter can be changed to be more generic if
   that means that code that called the original version can still be
   compiled.

-  Good exposure of the PTX should be high priority. If, at a new major
   version, we face a difficult choice between breaking
   backward-compatibility and an improvement of the PTX exposure, we
   will tend to the latter option more easily than in other parts of
   libhipcxx.

The API does not guarantee stability of template parameters. The order and
number of template parameters may change. Use arguments to driver overload
resolution as in the code below to ensure forward-compatibility:

.. code:: cuda

   // Use arguments to drive overload resolution:
   cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1);

   // Specifying templates directly is not forward-compatible, as order and number
   // of template parameters may change in a minor release:
   cuda::ptx::mbarrier_arrive_expect_tx<cuda::ptx::sem_release_t>(
     cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1
   );

**PTX ISA version and compute capability.** Each binding notes under
which PTX ISA version and SM version it may be used. Example:

.. code:: cuda

   // mbarrier.arrive.shared::cta.b64 state, [addr]; // 1.  PTX ISA 70, SM_80
   __device__ inline uint64_t mbarrier_arrive(
     cuda::ptx::sem_release_t sem,
     cuda::ptx::scope_cta_t scope,
     cuda::ptx::space_shared_t space,
     uint64_t* addr);

To check if the current compiler is recent enough, use:

.. code:: cuda

   #if __cccl_ptx_isa >= 700
   cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1);
   #endif

Ensure that you only call the function when compiling for a recent
enough compute capability (SM version), like this:

.. code:: cuda

   NV_IF_TARGET(NV_PROVIDES_SM_80,(
     cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1);
   ));

For more information on which compilers correspond to which PTX ISA, see
the `PTX ISA release notes <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes>`__.
