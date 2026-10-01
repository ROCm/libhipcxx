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
  :description: API reference for cuda::mr::resource_ref and cuda::mr::async_resource_ref, type-erased memory resource wrappers that reduce compile time while preserving property constraints in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, resource_ref, async_resource_ref, type erasure, memory resource, properties

.. _libcudacxx-extended-api-memory-resources-resource-ref:

``cuda::resource_ref``: a type-constrained resource wrapper
-----------------------------------------------------------

This page documents ``cuda::mr::resource_ref`` and ``cuda::mr::async_resource_ref``, type-erased memory resource wrappers that reduce compile time while preserving property constraints.

With the property design depicted in :ref:`cuda::get_property <libcudacxx-extended-api-memory-resources-properties>`,
a library has flexibility in checking constraints and querying custom properties. However, there is also a cost in
providing function templates for a potentially wide range of inputs. Depending on the number of different memory
resources, both compile time and binary size might increase considerably.

The type-erased ``resource_ref`` and ``async_resource_ref`` resource wrappers aid in efficiently coalescing such APIs
into a single function.

.. code:: cpp

   void* do_allocate_async(cuda::mr::async_resource_ref<> resource, std::size_t size, std::size_t align, cuda::stream_ref stream) {
       return resource.allocate_async(size, align, stream);
   }

   my_async_memory_resource resource;
   my_async_memory_resource* pointer_to_resource = &resource;

   void* from_reference = do_allocate_async(resource, 1337, 256, cuda::stream_ref{});
   void* from_ptr = do_allocate_async(pointer_to_resource, 1337, 256, cuda::stream_ref{});

Note that ``do_allocate_async`` is not a template anymore but a plain old function. The wrapper
``cuda::mr::{async_}resource_ref<>`` is constructible from any non-const reference or pointer to a memory resource that
satisfies ``cuda::mr::{async_}resource``.

Properties may also be passed to ``cuda::mr::{async_}resource_ref`` just as with ``cuda::mr::resource_with``.

.. code:: cpp

   struct required_alignment{};
   void* do_allocate_async_with_alignment(cuda::mr::async_resource_ref<required_alignment> resource, std::size_t size, cuda::stream_ref stream) {
       return resource.allocate_async(size, cuda::mr::get_property(resource, required_alignment), stream);
   }

However, the type erasure comes with the cost that arbitrary properties cannot be queried:

.. code:: cpp

   struct required_alignment{};
   void* buggy_allocate_async_with_alignment(cuda::mr::async_resource_ref<> resource, std::size_t size, cuda::stream_ref stream) {
       if constexpr (cuda::has_property<required_alignment>) { // BUG: This will always be false
           return resource.allocate_async(size, cuda::mr::get_property(resource, required_alignment), stream);
       } else {
           return resource.allocate_async(size, my_default_alignment, stream);
       }
   }

So, choose wisely. If your library has a well-defined set of fixed properties that you expect to always be available,
then ``cuda::mr::{async_}resource_ref`` is an amazing tool to improve compile times and binary size. If you need a
flexible interface then constraining through ``cuda::mr::{async_}resource_with`` is the proper solution.
