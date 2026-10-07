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
  :description: Documentation of the libhipcxx ThreadGroup concept, which defines the requirements for a type representing a group of cooperating threads in HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, ThreadGroup, thread group, cooperative groups, size, thread_rank, sync

.. _libcudacxx-extended-api-thread-groups:

Thread Groups
=============

This page documents the ThreadGroup concept, which defines the requirements for a type representing a group of cooperating threads in HIP.

.. code:: cpp

   struct ThreadGroup {
     static constexpr cuda::thread_scope thread_scope;
     Integral size() const;
     Integral thread_rank() const;
     void sync() const;
   };

The *ThreadGroup concept* defines the requirements of a type that represents a group of cooperating threads.

The `HIP cooperative groups API <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html>`_
provides a number of types that satisfy this concept.

Data Members
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Member
     - Description

   * - ``thread_scope``
     - The scope at which ``ThreadGroup::sync()`` synchronizes memory operations and thread execution.

Member Functions
----------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Function
     - Description

   * - ``size``
     - Returns the number of participating threads.
   * - ``thread_rank``
     - Returns a unique value for each participating thread (``0 <= ThreadGroup::thread_rank() < ThreadGroup::size()``).
   * - ``sync``
     - Synchronizes the participating threads.

.. note::

  This concept is defined for documentation purposes but is not materialized in the library.

Example
-------

.. code:: cpp

   #include <cuda/atomic>
   #include <cuda/std/cstddef>

   struct single_thread_group {
     static constexpr cuda::thread_scope thread_scope = cuda::thread_scope::thread_scope_thread;
     cuda::std::size_t size() const { return 1; }
     cuda::std::size_t thread_rank() const { return 0; }
     void sync() const {}
   };

..
   `See it on Godbolt <https://godbolt.org/z/6c16KxqY7>`_
