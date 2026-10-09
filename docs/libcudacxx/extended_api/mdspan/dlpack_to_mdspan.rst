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
  :description: API reference for cuda::to_host_mdspan, cuda::to_device_mdspan, and cuda::to_managed_mdspan, which convert a DLPack DLTensor to an mdspan in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, mdspan, DLPack, DLTensor, to_device_mdspan, to_host_mdspan, conversion

.. _libcudacxx-extended-api-mdspan-dlpack-to-mdspan:

DLPack to ``mdspan``
====================

This functionality provides a conversion from `DLPack <https://dmlc.github.io/dlpack/latest/>`__ ``DLTensor`` to ``cuda::host_mdspan``, ``cuda::device_mdspan``, and ``cuda::managed_mdspan``.

Defined in the ``<cuda/mdspan>`` header.

Conversion functions
--------------------

.. code:: cpp

   namespace cuda {

   template <typename ElementType, size_t Rank, typename LayoutPolicy = cuda::layout_stride_relaxed>
   [[nodiscard]] cuda::host_mdspan<ElementType, cuda::std::dims<Rank, int64_t>, LayoutPolicy>
   to_host_mdspan(const DLTensor& tensor);

   template <typename ElementType, size_t Rank, typename LayoutPolicy = cuda::layout_stride_relaxed>
   [[nodiscard]] cuda::device_mdspan<ElementType, cuda::std::dims<Rank, int64_t>, LayoutPolicy>
   to_device_mdspan(const DLTensor& tensor);

   template <typename ElementType, size_t Rank, typename LayoutPolicy = cuda::layout_stride_relaxed>
   [[nodiscard]] cuda::managed_mdspan<ElementType, cuda::std::dims<Rank, int64_t>, LayoutPolicy>
   to_managed_mdspan(const DLTensor& tensor);

   } // namespace cuda

Template parameters
-------------------

- ``ElementType``: The element type of the resulting ``mdspan``. Must match the ``DLTensor::dtype``.
- ``Rank``: The number of dimensions. Must match ``DLTensor::ndim``.
- ``LayoutPolicy``: The layout policy for the resulting ``mdspan``. Defaults to ``cuda::layout_stride_relaxed``. Supported layouts are:

  - ``cuda::std::layout_right`` (C-contiguous, row-major)
  - ``cuda::std::layout_left`` (Fortran-contiguous, column-major)
  - ``cuda::std::layout_stride`` (general strided layout)
  - ``cuda::layout_stride_relaxed`` (general strided layout with negative/zero strides and offset support)

Semantics
---------

The conversion produces a non-owning ``mdspan`` view of the ``DLTensor`` data:

- For ``layout_right``, ``layout_left``, and ``layout_stride``, the data pointer is computed as ``static_cast<char*>(tensor.data) + tensor.byte_offset``.
- For ``layout_stride_relaxed``, the data pointer is ``tensor.data`` directly (no ``byte_offset`` adjustment). Instead, ``tensor.byte_offset`` is converted to an element offset (``byte_offset / sizeof(ElementType)``) and stored in the mapping. This ensures that ``mapping(indices...) = offset + sum(index_i * stride_i)`` produces non-negative indices even with negative strides, and ``required_span_size()`` correctly reflects the actual memory span.
- For ``rank > 0``, ``mdspan.extent(i)`` is ``tensor.shape[i]``.
- For ``layout_stride`` and ``layout_stride_relaxed``, ``mdspan.stride(i)`` is ``tensor.strides[i]`` (or computed as row-major if ``strides`` is ``nullptr`` for DLPack < v1.2).
- The device type is validated:

  - ``kDLCPU`` for ``to_host_mdspan``
  - ``kDLCUDA`` for ``to_device_mdspan``
  - ``kDLCUDAManaged`` for ``to_managed_mdspan``

Supported element types:

- ``bool``.
- Signed and unsigned integers.
- IEEE-754 Floating-point and extended precision floating-point, including ``__half``, ``__nv_bfloat16``, ``__float128``, FP8, FP6, FP4 when available.
- Complex: ``cuda::std::complex<__half>``, ``cuda::std::complex<float>``, and ``cuda::std::complex<double>``.
- `HIP built-in vector types <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html>`__, such as ``int2``, ``float4``, etc.
- Vector types for extended floating-point, such as ``__half2``, ``__nv_fp8x4_e4m3``, etc.

Constraints
-----------

- ``LayoutPolicy`` must be one of ``cuda::std::layout_right``, ``cuda::std::layout_left``, ``cuda::std::layout_stride``, or ``cuda::layout_stride_relaxed``.
- For ``layout_right`` and ``layout_left``, the ``DLTensor`` strides must be compatible with the layout.

Runtime errors
--------------

The conversion throws ``std::invalid_argument`` in the following cases:

- ``DLTensor::ndim`` does not match the specified ``Rank``.
- ``DLTensor::dtype`` does not match ``ElementType``.
- ``DLTensor::data`` is ``nullptr``.
- ``DLTensor::shape`` is ``nullptr`` (for rank > 0).
- Any ``DLTensor::shape[i]`` is negative.
- ``DLTensor::strides`` is ``nullptr`` for DLPack v1.2 or later.
- ``DLTensor::strides`` is ``nullptr`` for ``layout_left`` with rank > 1 (DLPack < v1.2).
- ``DLTensor::strides[i]`` is not positive for ``layout_stride``.
- ``DLTensor::strides`` are not compatible with the requested ``layout_right`` or ``layout_left``.
- ``DLTensor::byte_offset`` is not a multiple of the element size for ``layout_stride_relaxed``.
- ``DLTensor::device.device_type`` does not match the target mdspan type.
- Data pointer is not properly aligned for the element type.

Availability notes
------------------

- This API is available only when DLPack header is present, namely ``<dlpack/dlpack.h>`` is found in the include path.
- This API can be disabled by defining ``CCCL_DISABLE_DLPACK`` before including any library headers. In this case, ``<dlpack/dlpack.h>`` will not be included.

References
----------

- `DLPack C API <https://dmlc.github.io/dlpack/latest/c_api.html>`__ documentation.

Example
-------

.. code:: cpp

  #include <dlpack/dlpack.h>
  #include <cuda/mdspan>
  #include <cuda/std/cassert>
  #include <cuda/std/cstdint>

  int main() {
    int data[6] = {0, 1, 2, 3, 4, 5};

    // Create a DLTensor manually for demonstration
    int64_t shape[2]   = {2, 3};
    int64_t strides[2] = {3, 1};  // row-major strides

    DLTensor tensor{};
    tensor.data        = data;
    tensor.device      = {kDLCPU, 0};
    tensor.ndim        = 2;
    tensor.dtype       = DLDataType{kDLInt, 32, 1};
    tensor.shape       = shape;
    tensor.strides     = strides;
    tensor.byte_offset = 0;

    // Convert to host_mdspan
    auto md = cuda::to_host_mdspan<int, 2>(tensor);

    assert(md.rank() == 2);
    assert(md.extent(0) == 2 && md.extent(1) == 3);
    assert(md.stride(0) == 3 && md.stride(1) == 1);
    assert(md.data_handle() == data);
    assert(md(0, 0) == 0 && md(1, 2) == 5);
  }

See also
--------

- :ref:`libcudacxx-extended-api-mdspan-mdspan-to-dlpack` for the reverse conversion.
