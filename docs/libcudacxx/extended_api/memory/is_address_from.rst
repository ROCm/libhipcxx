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
  :description: API reference for cuda::device::is_address_from and cuda::device::is_object_from, which check whether a pointer or object belongs to a given device address space in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, is_address_from, is_object_from, address_space, shared memory, global memory

.. _libcudacxx-extended-api-memory-is_address_from:

``cuda::device::is_address_from`` and ``cuda::device::is_object_from``
======================================================================

This page documents ``cuda::device::is_address_from`` and ``cuda::device::is_object_from``, which check whether a pointer or object belongs to a specific device address space.

Defined in the ``<cuda/memory>`` header.

.. code:: cpp

   namespace cuda::device {

   enum class address_space
   {
     global,         // Global state space
     shared,         // Shared state space
     constant,       // Constant state space
     local,          // Local state space
     grid_constant,  // Kernel function parameter in the parameter state space
     cluster_shared, // Cluster shared window within the shared state space
   };

   } // namespace cuda::device

Enumeration of device address spaces used with the ``is_address_from()`` and ``is_object_from()`` functions.

..
   See the `PTX ISA documentation for state spaces <https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces>`_ for more details.

----

.. code:: cpp

   namespace cuda::device {

   [[nodiscard]] __device__ inline
   bool is_address_from(const volatile void* ptr, address_space space) noexcept; // (1)

   } // namespace cuda::device

Checks whether a generic-address pointer ``ptr`` is from the specified address space.

----

.. code:: cpp

   namespace cuda::device {

   template <typename T>
   [[nodiscard]] __device__ inline
   bool is_object_from(T& obj, address_space space) noexcept; // (2)

   } // namespace cuda::device

Checks whether an object ``obj`` with a generic address is from the specified address space.

----

..
   Unlike the corresponding CUDA intrinsic functions ``__isGlobal()``, ``__isShared()``, ``__isConstant()``, ``__isLocal()``, ``__isGridConstant()``, and ``__isClusterShared()``, ``is_address_from()`` and ``is_object_from()`` are portable across all compute capabilities and, in debug mode, also checks that the pointer is not null.

In debug mode, ``is_address_from()`` and ``is_object_from()`` also check that the pointer is not null.

**Parameters**

- ``ptr``: The pointer. (1)
- ``obj``: The object. (2)
- ``space``: The address space. (1, 2)

**Return value**

- Returns ``true`` if the pointer (1) or object (2) is from the specified address space; ``false`` otherwise.

.. note::

  If the GPU architecture does not support the requested address space, the function always returns ``false``.

  On AMD GPUs, ``address_space::grid_constant`` queries always return ``false``, and ``address_space::cluster_shared``
  queries are equivalent to ``address_space::shared`` queries. ``address_space::constant`` queries are not supported:
  ``__constant__`` variables cannot be distinguished from global memory, and the query traps at run time.

**Preconditions**

- ``ptr`` must not be null. (1)

**Performance considerations**

- When available, the built-in functions (``__isGlobal()``, ``__isShared()``, ``__isConstant()``, ``__isLocal()``, ``__isGridConstant()``, or ``__isClusterShared()``) are used to determine the address space.
- If the memory space of the input pointer matches the requested address space,
  the function marks the pointer as belonging to that address space.

..
   For example, a subsequent store to a generic address that maps to shared memory emits an ``STS`` SASS instruction rather than the generic ``ST`` instruction.

Example
-------

.. code:: cpp

    #include <cuda/memory>

    __device__   int global_var;

    __global__ void kernel()
    {
        using cuda::device::address_space;
        __shared__ int shared_var;
        int local_var{};

        assert(cuda::device::is_address_from(&global_var, address_space::global));
        assert(cuda::device::is_address_from(&shared_var, address_space::shared));
        assert(cuda::device::is_address_from(&local_var, address_space::local));

        assert(cuda::device::is_object_from(global_var, address_space::global));
        assert(cuda::device::is_object_from(shared_var, address_space::shared));
        assert(cuda::device::is_object_from(local_var, address_space::local));
    }

    int main(int, char**)
    {
        kernel<<<1, 1>>>();
        hipDeviceSynchronize();
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/5ajhe37df>`__
