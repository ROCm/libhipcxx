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
  :description: API reference for cuda::access_property, an opaque type encoding memory access properties such as normal, persisting, and streaming cache eviction hints in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, access_property, cache eviction, persisting, streaming, global memory, L2 cache

.. _libcudacxx-extended-api-memory-access-properties-access-property:

``cuda::access_property``
=========================

This page documents ``cuda::access_property``, an opaque type encoding memory access properties such as normal, persisting, and streaming cache eviction hints.

Defined in header ``<cuda/annotated_ptr>``:

.. code:: cuda

   namespace cuda {
   class access_property;
   } // namespace cuda

The class ``cuda::access_property`` is a `LiteralType <https://en.cppreference.com/w/cpp/named_req/LiteralType>`_
that provides an opaque encoding for properties of memory operations. It is used in combination with
:ref:`cuda::annotated_ptr <libcudacxx-extended-api-memory-access-properties-annotated-ptr>`,
:ref:`cuda::associate_access_property <libcudacxx-extended-api-memory-access-properties-associate-access-property>` and
:ref:`cuda::apply_access_property <libcudacxx-extended-api-memory-access-properties-apply-access-property>`
to *request* the application of properties to memory operations.

.. code:: cuda

   namespace cuda {

   class access_property {
     public:
       // Static memory space property:
       struct shared {};
       struct global {};

       // Static global memory residence control property:
       struct normal {
           __host__ __device__ constexpr operator hipAccessProperty() const noexcept;
       };
       struct persisting {
           __host__ __device__ constexpr operator hipAccessProperty() const noexcept;
       };
       struct streaming {
           __host__ __device__ constexpr operator hipAccessProperty() const noexcept;
       };

       // Default constructor:
       __host__ __device__ constexpr access_property() noexcept;

       // Copy constructor:
       constexpr access_property(access_property const&) noexcept = default;

       // Copy assignment:
       access_property& operator=(const access_property& other) noexcept = default;

       // Constructors from static global memory residence control properties:
       __host__ __device__ constexpr access_property(global)     noexcept;
       __host__ __device__ constexpr access_property(normal)     noexcept;
       __host__ __device__ constexpr access_property(streaming)  noexcept;
       __host__ __device__ constexpr access_property(persisting) noexcept;

       // Dynamic interleaved global memory residence control property constructors:
       __host__ __device__ constexpr access_property(normal,     float probability);
       __host__ __device__ constexpr access_property(streaming,  float probability);
       __host__ __device__ constexpr access_property(persisting, float probability);
       __host__ __device__ constexpr access_property(normal,     float probability, streaming);
       __host__ __device__ constexpr access_property(persisting, float probability, streaming);

       // Dynamic range global memory residence control property constructors:
       __host__ __device__ constexpr access_property(void* ptr, size_t partition_bytes, size_t total_bytes, normal);
       __host__ __device__ constexpr access_property(void* ptr, size_t partition_bytes, size_t total_bytes, streaming);
       __host__ __device__ constexpr access_property(void* ptr, size_t partition_bytes, size_t total_bytes, persisting);
       __host__ __device__ constexpr access_property(void* ptr, size_t partition_bytes, size_t total_bytes, normal,     streaming);
       __host__ __device__ constexpr access_property(void* ptr, size_t partition_bytes, size_t total_bytes, persisting, streaming);
   };

   } // namespace cuda

Kinds of access properties
--------------------------

Access properties are either *static* compile-time values or *dynamic* runtime values. The following properties
of a memory access are provided:

-  Static memory space properties:

   .. _libcudacxx-extended-api-memory-access-properties-access-property-shared:

   -  ``cuda::access_property::shared``: memory access to the shared memory space,

-  Static global memory space *and* residence control properties:

   .. _libcudacxx-extended-api-memory-access-properties-access-property-global:

   -  ``cuda::access_property::global``: memory access to the global memory space without indicating an expected
      frequency of access to that memory,

   .. _libcudacxx-extended-api-memory-access-properties-access-property-normal:

   -  ``cuda::access_property::normal``: memory access to the global memory space expecting the memory to be
      accessed as frequent as other memory,

   .. _libcudacxx-extended-api-memory-access-properties-access-property-persisting:

   -  ``cuda::access_property::persisting``: memory access to the global memory space expecting the memory to be
      accessed more frequently than other memory; this priority is suitable for data that should remain persistent in cache,

   .. _libcudacxx-extended-api-memory-access-properties-access-property-streaming:

   -  ``cuda::access_property::streaming``: memory access to the global memory space expecting the memory to be
      accessed infrequently; this priority is suitable for streaming data.

-  Dynamic global memory residence control properties:

   -  ``normal``, ``persisting``, ``streaming``: static memory residence control properties may be specified at runtime,
   -  ``interleaved``: choose a ``probability`` of memory addresses to be accessed with one property and the remaining
      ``1 - probability`` addresses with another,
   -  ``range``: choose a partitioned memory range with memory accesses to the "middle" sub-partition using the
      *primary* property, and memory accesses to the head and tail sub-partitions using the *secondary* property.

**Note**: The difference between ``cuda::access_property::global`` and ``cuda::access_property::normal`` is subtle.
The ``cuda::access_property::normal`` hints that the pointer points to the global address space *and* the memory will
be accessed with "normal frequency", while ``cuda::access_property::global`` only hints that the pointer points to
the global address-space, it does not hint about how frequent the accesses will be.

.. warning::

   The behavior of *requesting* the application of ``cuda::access_property`` to memory accesses, or their association
   with memory addresses, outside of the corresponding address space is *undefined*
   (note: even if that address is not "used").

Default constructor
-------------------

.. code:: cuda

   __host__ __device__ constexpr access_property() noexcept;

**Effects**: as if ``access_property(global)``.

Static global memory residence control property constructors
------------------------------------------------------------

.. code:: cuda

   __host__ __device__ constexpr access_property::access_property(global) noexcept;
   __host__ __device__ constexpr access_property::access_property(normal) noexcept;
   __host__ __device__ constexpr access_property::access_property(streaming) noexcept;
   __host__ __device__ constexpr access_property::access_property(persisting) noexcept;

**Effects**: as-if ``access_property(PROPERTY, 1.0)`` where ``PROPERTY``
is one of ``global``, ``normal``, ``streaming``, or ``persisting``.

Dynamic interleaved global memory residence control property constructors
-------------------------------------------------------------------------

.. code:: cuda

   __host__ __device__ constexpr access_property::access_property(normal,     float probability);
   __host__ __device__ constexpr access_property::access_property(streaming,  float probability);
   __host__ __device__ constexpr access_property::access_property(persisting, float probability);
   __host__ __device__ constexpr access_property::access_property(normal,     float probability, streaming);
   __host__ __device__ constexpr access_property::access_property(persisting, float probability, streaming);

**Preconditions**: ``0 < probability <= 1.0``.

**Effects**: constructs an *interleaved* access property that *requests*
the first and third arguments - access properties - to be applied with
``probability`` and ``1 - probability`` to memory accesses. The
overloads without a third argument request applying ``global`` with
``1 - probability``.

Dynamic range global memory residence control property constructors
-------------------------------------------------------------------

.. code:: cuda

   __host__ __device__ constexpr access_property::access_property(void* ptr, size_t leading_bytes, size_t total_bytes, normal);
   __host__ __device__ constexpr access_property::access_property(void* ptr, size_t leading_bytes, size_t total_bytes, streaming);
   __host__ __device__ constexpr access_property::access_property(void* ptr, size_t leading_bytes, size_t total_bytes, persisting);
   __host__ __device__ constexpr access_property::access_property(void* ptr, size_t leading_bytes, size_t total_bytes, normal,     streaming);
   __host__ __device__ constexpr access_property::access_property(void* ptr, size_t leading_bytes, size_t total_bytes, persisting, streaming);

..

   note: pointer arithmetic below performed ``char* ptr`` instead of
   ``void* ptr``

**Preconditions**:

- ``ptr`` is a generic pointer that is *valid* to cast to a pointer to the global memory address space.
- ``0 < leading_bytes <= total_bytes <= 4GB``.

**Postconditions**: memory accesses requesting the application of this
property must be in range
``[max(0, ptr + leading_bytes - total_bytes), ptr + total_bytes)``.

**Effects**: the fourth and fifth arguments, access properties, are
called *primary* and *secondary* properties. The overloads without a
fifth argument use ``global`` as the *secondary* property. Constructs a
*range* access property *requesting* the properties to be
approximately applied to memory accesses as follows:

-  secondary property to accesses in address-range:
   ``[max(0, ptr + leading_bytes - total_bytes), ptr)``
-  primary property to accesses in address-range:
   ``[ptr, ptr + leading_bytes)``
-  secondary property to accesses in address-range:
   ``[ptr + leading_bytes, ptr + total_bytes)``

**Note**: This property enables three main use cases:

1. Unary range ``[ptr, ptr + total_bytes)`` with primary property by
   using ``leading_bytes == total_bytes``.

2. Binary range ``[ptr, ptr + leading_bytes)`` and
   ``[ptr + leading_bytes, ptr + total_bytes)`` with primary and
   secondary properties by just not using this range to access any
   memory in range ``[max(0, ptr + leading_bytes - total_bytes), ptr)``.

3. Primary range with secondary "halo" ranges (see example below). Given
   ``leading_bytes`` for the primary range, and ``halo_bytes`` for the
   size of each of the secondary ranges by using
   ``total_bytes == leading_bytes + halo_bytes``:

   .. code:: cpp

       ____________________________________________________________
      |  halo / secondary | leading / primary   | halo / secondary |
       ------------------------------------------------------------
                          ^
                          | ptr

      |<-- halo_bytes  -->|<-- leading_bytes -->|<-- halo_bytes -->|
                          |<--            total_bytes           -->|

Conversion operators
--------------------

.. code:: cuda

   __host__ __device__ constexpr access_property::normal::operator hipAccessProperty() const noexcept;
   __host__ __device__ constexpr access_property::streaming::operator hipAccessProperty() const noexcept;
   __host__ __device__ constexpr access_property::persisting::operator hipAccessProperty() const noexcept;

**Returns**: corresponding HIP runtime
`hipAccessProperty <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___global_defs.html>`_
value.

**Note**: Allows ``constexpr cuda::access_property::normal{}``,
``cuda::access_property::streaming{}``, and
``cuda::access_property::persisting{}`` to be used in lieu of the
corresponding HIP runtime
`hipAccessProperty <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___global_defs.html>`_
enumerated values.

Example
-------

.. code:: cuda

   #include <cuda/annotated_ptr>

   __global__ void undefined_behavior(int* global) {
       // Associating pointers with mismatching address spaces is undefined:
       cuda::associate_access_property(global, cuda::access_property::shared{});     // undefined behavior
       __shared__ int shmem;
       cuda::associate_access_property(&shmem, cuda::access_property::normal{});     // undefined behavior
       cuda::associate_access_property(&shmem, cuda::access_property::streaming{});  // undefined behavior
       cuda::associate_access_property(&shmem, cuda::access_property::persisting{}); // undefined behavior

       cuda::access_property interleaved_implicit_global(cuda::access_property::streaming{}, 0.5);
       cuda::associate_access_property(&shmem, interleaved_implicit_global);         // undefined behavior

       cuda::access_property range_implicit_global0(&shmem, 0, sizeof(int), cuda::access_property::streaming{});
       cuda::associate_access_property(&shmem, range_implicit_global0);              // undefined behavior

       // Using a zero probability or probability out-of-range (0, 1] is undefined:
       cuda::access_property interleaved(cuda::access_property::streaming{}, 0.0);   // undefined behavior
   }

   __global__ void correct(int* global) {
       __shared__ int shmem;
       cuda::associate_access_property(&shmem, cuda::access_property::shared{});

       cuda::access_property global_range0(global, 0, sizeof(int), cuda::access_property::streaming{});
       cuda::associate_access_property(global, global_range0);

       cuda::access_property global_interleaved(cuda::access_property::streaming{}, 1.0);
       cuda::associate_access_property(global, global_interleaved);

       // Access properties can be constructed for any address range
       cuda::access_property global_range1(global,  0, sizeof(int), cuda::access_property::streaming{});
       cuda::access_property global_range2(nullptr, 0, sizeof(int), cuda::access_property::streaming{});
   }

   __global__ void range(int* g, size_t n) {
     // To apply a single property to all elements in the range [g, g+n), set leading_bytes = total_bytes = n
     auto range_property = cuda::access_property(g, n, n, cuda::access_property::persisting{});
   }

   __global__ void range_with_halos(int* g, size_t n, size_t halos) {
       // In the range [g, g + n), the first and last "halos" elements of `int` type are halos.
       // This example applies one property to the halo elements, and another property to the internal elements:
       // - halos: streaming  (secondary property)
       // - internal: persisting (primary property)

       auto internal_property = cuda::access_property::persisting{};
       auto halo_property = cuda::access_property::streaming{};

       // For the range property, the pointer used to build the property
       // must satisfy p = g + halos
       int* p = g + halos;
       // Then, "total_elements" (total_size * sizeof(int)) must satisfy:
       // g + n = p + total_elements
       int total_bytes = (g + n - p) * sizeof(int);
       // Finally, "leading_elements" (leading_bytes * sizeof(int)) must satisfy:
       // g = p + leading_elements - total_elements
       int leading_bytes = (g - p) * sizeof(int) + total_bytes;

       // Is a property that we can use for halo exchange:
       auto range_property = cuda::access_property(p, leading_bytes, total_bytes, internal_property, halo_property);
   }
