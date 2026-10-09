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
  :description: Description of the tuple protocol support for HIP vector types in libhipcxx, enabling structured bindings and cuda::std::get access.
  :keywords: libhipcxx, ROCm, HIP, C++, tuple protocol, vector types, structured bindings, tuple_size, get

.. _libcudacxx-extended-api-vector-tuple-protocol:

Vector Tuple Protocol
=====================

Defined in the ``<cuda/std/tuple>`` header.

``libhipcxx`` provides tuple protocol support for HIP vector types, enabling structured bindings and tuple-like access to vector type elements.

Please refer to the documentation of the C++ standard header `\<tuple\> <https://en.cppreference.com/w/cpp/header/tuple>`_ for more information.

- ``cuda::std::tuple_size<VectorType>`` provides the number of elements in a HIP vector type as a compile-time constant.
- ``cuda::std::tuple_element<I, VectorType>`` provides the scalar element type of a HIP vector type at index ``I``.
- ``cuda::std::get<I>(v)`` returns a reference to the element at index ``I`` of the vector type ``v``.

  - For ``I == 0``: reference to ``v.x``
  - For ``I == 1``: reference to ``v.y``
  - For ``I == 2``: reference to ``v.z``
  - For ``I == 3``: reference to ``v.w``

Structured Bindings
-------------------

HIP vector types support C++17 structured bindings through the tuple protocol.

.. code:: cpp

   int2 vec{1, 2};
   auto [x, y] = vec;  // x == 1, y == 2

   float4 vec4{1.0f, 2.0f, 3.0f, 4.0f};
   auto& [a, b, c, d] = vec4;  // references to vec4.x, vec4.y, vec4.z, vec4.w

Supported Vector Types
----------------------

The tuple protocol is supported for all HIP vector types:

**Integral vector types**

- ``char1``, ``char2``, ``char3``, ``char4``.
- ``uchar1``, ``uchar2``, ``uchar3``, ``uchar4``.
- ``short1``, ``short2``, ``short3``, ``short4``.
- ``ushort1``, ``ushort2``, ``ushort3``, ``ushort4``.
- ``int1``, ``int2``, ``int3``, ``int4``.
- ``uint1``, ``uint2``, ``uint3``, ``uint4``.
- ``long1``, ``long2``, ``long3``, ``long4``.
- ``ulong1``, ``ulong2``, ``ulong3``, ``ulong4``.
- ``longlong1``, ``longlong2``, ``longlong3``, ``longlong4``.
- ``ulonglong1``, ``ulonglong2``, ``ulonglong3``, ``ulonglong4``.

**Floating-point vector types**

- ``float1``, ``float2``, ``float3``, ``float4``.
- ``double1``, ``double2``, ``double3``, ``double4``.

**Special types**

- ``dim3``.

**Extended floating-point vector types**

- ``__half2``.
- ``__nv_bfloat162``.
- ``__nv_fp8x2_e4m3``, ``__nv_fp8x2_e5m2``, ``__nv_fp8x4_e4m3``, ``__nv_fp8x4_e5m2`` **\***.
- ``__nv_fp8x2_e8m0``, ``__nv_fp8x4_e8m0``  **\***.
- ``__nv_fp6x2_e2m3``, ``__nv_fp6x2_e3m2``, ``__nv_fp6x4_e2m3``, ``__nv_fp6x4_e3m2``  **\***.
- ``__nv_fp4x2_e2m1``, ``__nv_fp4x4_e2m1``  **\***.

**\*** Single-component vector types.

.. note::

   With HIP, the ``e8m0`` FP8 types and the FP6 and FP4 vector types are not available.

Example
-------

.. code:: cpp

   #include <cuda/std/tuple>

   __host__ __device__ void example() {
       // tuple_size: get the number of elements
       static_assert(cuda::std::tuple_size_v<int2>   == 2);
       static_assert(cuda::std::tuple_size_v<float4> == 4);
       static_assert(cuda::std::tuple_size_v<dim3>   == 3);

       // tuple_element: get the element type
       static_assert(cuda::std::is_same_v<cuda::std::tuple_element_t<0, int2>, int>);
       static_assert(cuda::std::is_same_v<cuda::std::tuple_element_t<1, float4>, float>);
       static_assert(cuda::std::is_same_v<cuda::std::tuple_element_t<2, dim3>, unsigned int>);

       // get<>: access elements by index
       int2 vec2{10, 20};
       int x = cuda::std::get<0>(vec2);  // x == 10
       int y = cuda::std::get<1>(vec2);  // y == 20

       // Modify elements through get<>
       cuda::std::get<0>(vec2) = 100;    // vec2.x == 100

       // Structured bindings (C++17)
       float3 color{0.5f, 0.8f, 1.0f};
       auto [r, g, b] = color;           // r == 0.5f, g == 0.8f, b == 1.0f

       // Reference bindings for modification
       int4 data{1, 2, 3, 4};
       auto& [a, b2, c, d] = data;
       a = 10;                           // data.x == 10
   }

See Also
--------

- :ref:`Type Traits for HIP Vector Types <libcudacxx-extended-api-type_traits-vector_types>` - Extract scalar type from vector types.
