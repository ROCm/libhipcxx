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
  :description: API reference for cuda::restrict_accessor and cuda::restrict_mdspan, which apply the restrict aliasing policy to mdspan accessors in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, restrict_accessor, restrict_mdspan, aliasing policy, mdspan, __restrict__

.. _libcudacxx-extended-api-mdspan-restrict-accessor:

``restrict`` ``mdspan`` and ``accessor``
========================================

This page documents ``cuda::restrict_accessor`` and ``cuda::restrict_mdspan``, which apply the restrict aliasing policy to mdspan accessors for improved compiler optimization.

.. code:: cpp

  template <typename Accessor>
  using restrict_accessor;

An alias type to create an accessor with the *restrict aliasing policy* starting from an existing accessor.

More information related to the *restrict aliasing policy* can be found in the HIP documentation: `__restrict__ keyword <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#restrict>`_.

----

.. code:: cpp

  template <typename ElementType,
            typename Extents,
            typename LayoutPolicy   = cuda::std::layout_right,
            typename AccessorPolicy = cuda::std::default_accessor<_ElementType>>
  using restrict_mdspan = cuda::std::mdspan<ElementType, Extents, LayoutPolicy, restrict_accessor<AccessorPolicy>>;

An alias type to create an ``mdspan`` with a *restrict aliasing policy* accessor.

----

Traits:

.. code:: cpp

    template <typename T>
    inline constexpr bool is_restrict_accessor_v = /*true if T is a restrict accessor, false otherwise*/;

    template <typename T>
    inline constexpr bool is_restrict_mdspan_v = /*true if T is a restrict mdspan, false otherwise*/;

----

**Constraints**:

- Accessor ``data_handle_type`` must be a pointer type.

Example
-------

.. code:: cpp

    #include <cuda/mdspan>

    using restrict_mdspan = cuda::restrict_mdspan<int, cuda::std::dims<1>>;

    __host__ __device__ void
    compute(restrict_mdspan a, restrict_mdspan b, restrict_mdspan c) {
        c[0] = a[0] * b[0];
        c[1] = a[0] * b[0];
        c[2] = a[0] * b[0] * a[1];
        c[3] = a[0] * a[1];
        c[4] = a[0] * b[0];
        c[5] = b[0];
    }

    int main() {
        using  dim      = cuda::std::dims<1>;
        using  mdspan   = cuda::std::mdspan<int, dim>;
        int    arrayA[] = {1, 2};
        int    arrayB[] = {5};
        int    arrayC[] = {9, 10, 11, 12, 13, 14};
        mdspan mdA{arrayA, dim{1}};
        mdspan mdB{arrayB, dim{5}};
        mdspan mdC{arrayC, dim{6}};
        compute(mdA, mdB, mdC);

        using restrict_aligned_accesor = cuda::std::restrict_accessor<cuda::std::aligned_accessor<int, 8>>;
        using restrict_aligned_mdspan  = cuda::std::mdspan<int, dim, layout_right, restrict_aligned_accesor>;
        restrict_aligned_mdspan mdD{mdC};
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/Wjco996z8>`_
