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
  :description: API reference for cuda::hierarchy, cuda::make_hierarchy and the thread hierarchy level descriptors and queries in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, thread hierarchy, grid, block, make_hierarchy, launch dimensions

.. _cccl-runtime-hierarchy:

.. |cuda_hierarchy| replace:: ``cuda::hierarchy``

.. |cuda_make_hierarchy| replace:: ``cuda::make_hierarchy``
.. |cuda_make_config| replace:: ``cuda::make_config``
.. |cuda_grid_dims| replace:: ``cuda::grid_dims``
.. |cuda_cluster_dims| replace:: ``cuda::cluster_dims``
.. |cuda_block_dims| replace:: ``cuda::block_dims``
.. |cuda_warp| replace:: ``cuda::warp``
.. |cuda_gpu_thread| replace:: ``cuda::gpu_thread``
.. |cuda_hierarchy_add_level| replace:: ``cuda::hierarchy_add_level``
.. |cuda_get_launch_dimensions| replace:: ``cuda::get_launch_dimensions``

Hierarchy
=========

The hierarchy API provides abstractions for representing and querying levels in the HIP thread hierarchy (grid,
block, warp, and thread levels). It enables compile-time and runtime queries of thread dimensions and counts across
different hierarchy levels.
See the `HIP programming model <https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html>`__
for an introduction to the thread hierarchy.

.. note::

   The API also defines a cluster level (``cuda::cluster``, |cuda_cluster_dims|) for thread block clusters.
   Thread block clusters are not supported on AMD GPUs: cluster dimensions are ignored when launching a kernel,
   so do not use the cluster level in libhipcxx code.

|cuda_hierarchy|
---------------------------------------------------------------------
.. _cccl-runtime-hierarchy-hierarchy:

|cuda_hierarchy| is a type representing a hierarchy of HIP threads. It combines hierarchy level descriptors
to represent dimensions of a (possibly partial) hierarchy. It supports accessing individual levels and queries
combining dimensions of multiple levels.

A hierarchy should be created using |cuda_make_hierarchy| rather than being constructed directly. The
hierarchy type can be used by itself, but its main purpose is to be part of a kernel launch configuration described
here: :ref:`Launch <cccl-runtime-launch>`. In that case, instead of calling |cuda_make_hierarchy|, the same arguments
can be passed to |cuda_make_config|.

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   auto h = cuda::make_hierarchy(
     cuda::grid_dims(256),
     cuda::block_dims<8, 8, 8>()
   );

   // Access level dimensions
   assert(h.level(cuda::grid).dims.x == 256);

   // Query counts across levels
   static_assert(cuda::gpu_thread.count(cuda::block, h) == 8 * 8 * 8);

|cuda_make_hierarchy|
----------------------------------------------------------------------------------------------------
.. _cccl-runtime-hierarchy-make-hierarchy:

|cuda_make_hierarchy| creates a hierarchy from passed hierarchy level descriptors. Levels can be passed in
ascending or descending order, and the function will automatically order them correctly.

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   // Levels can be passed in any order
   auto h1 = cuda::make_hierarchy(
     cuda::grid_dims(256),
     cuda::block_dims<8, 8, 8>()
   );

   auto h2 = cuda::make_hierarchy(
     cuda::block_dims<8, 8, 8>(),
     cuda::grid_dims(256)
   );

   // Both create equivalent hierarchies
   static_assert(cuda::std::is_same_v<decltype(h1), decltype(h2)>);

Hierarchy Level Descriptors
----------------------------
.. _cccl-runtime-hierarchy-level-descriptors:

The hierarchy API provides level descriptor functions for grid and block levels.
Each level supports both compile-time and runtime dimensions:

- |cuda_grid_dims| (compile-time and runtime overload forms)
- |cuda_block_dims| (compile-time and runtime overload forms)

..
   Thread block clusters are an NVIDIA-specific feature and are not supported on AMD GPUs.

   - |cuda_cluster_dims| (compile-time and runtime overload forms)

Warp and thread levels are implicit and are queried via level objects (e.g., |cuda_warp|,
|cuda_gpu_thread|).

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   auto h = cuda::make_hierarchy(
     cuda::grid_dims(256, 128),      // Runtime grid dimensions
     cuda::block_dims<32, 16>()      // Compile-time block dimensions
   );

Hierarchy Queries
-----------------
.. _cccl-runtime-hierarchy-queries:

Hierarchies support various query operations via level objects (``cuda::grid``,
``cuda::block``, |cuda_warp|, |cuda_gpu_thread|):

- ``unit.count(level, hierarchy)`` - Count units within a level (e.g., threads per block)
- ``unit.rank(level, hierarchy)`` - Get the rank (linear index) of a unit within a level (device only)
- ``unit.dims(level, hierarchy)`` - Get dimensions of units within a level
- ``hierarchy.level<Level>()`` - Get the level descriptor for a specific level
- ``hierarchy.fragment<Unit, Level>()`` - Extract a fragment of the hierarchy

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   auto h = cuda::make_hierarchy(
     cuda::grid_dims(256),
     cuda::block_dims<8, 8, 8>()
   );

   // Get block-level descriptor
   auto block_desc = h.level(cuda::block);
   assert(block_desc.dims.x == 8);

   // Count threads per block
   static_assert(cuda::gpu_thread.count(cuda::block, h) == 512);

   // Get fragment (block to grid)
   auto fragment = h.fragment(cuda::block, cuda::grid);

|cuda_hierarchy_add_level|
---------------------------------------------------------------------------------------------------------
.. _cccl-runtime-hierarchy-add-level:

|cuda_hierarchy_add_level| returns a new hierarchy that is a copy of the supplied hierarchy with a new level
added. The function automatically determines whether to add the level at the top or bottom based on the existing
levels.

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   auto partial = cuda::make_hierarchy<cuda::block_level>(
     cuda::grid_dims(256)
   );

   auto complete = cuda::hierarchy_add_level(
     partial,
     cuda::block_dims<8, 8, 8>()
   );

|cuda_get_launch_dimensions|
-----------------------------------------------------------------------------------------------------------
.. _cccl-runtime-hierarchy-launch-dimensions:

|cuda_get_launch_dimensions| returns a tuple of ``hierarchy_query_result`` objects containing dimensions from
the hierarchy that can be used to launch kernels. For a hierarchy without a cluster level, the returned tuple has two
elements (grid, block dimensions).

..
   Thread block clusters are an NVIDIA-specific feature and are not supported on AMD GPUs.

   The returned tuple has three elements if cluster_level is present
   (grid, cluster, block dimensions), or two elements otherwise (grid, block dimensions).

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   auto h = cuda::make_hierarchy(
     cuda::grid_dims(256),
     cuda::block_dims<8, 8, 8>()
   );

   auto [grid_dims, block_dims] = cuda::get_launch_dimensions(h);
   // Can be used with hipLaunchKernel or similar APIs
