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
  :description: API reference for cuda::device::warp_match_all, which checks whether all lanes in a warp hold the same value in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, warp, warp_match_all, lane_mask, warp intrinsics

.. _libcudacxx-extended-api-warp-warp-match-all:

``cuda::device::warp_match_all``
================================

This page documents ``cuda::device::warp_match_all``, which checks whether all lanes of a warp have the same value.

Defined in ``<cuda/warp>`` header.

.. code:: cpp

    namespace cuda::device {

    template <typename T>
    [[nodiscard]] __device__ bool
    warp_match_all(const T& data, lane_mask = lane_mask::all());

    } // namespace cuda::device

The functionality provides a generalized and safe alternative to the warp match all intrinsic ``__match_all_sync``.
The function allows bitwise comparison of any data size, including raw arrays, pointers, and structs.

**Parameters**

- ``data``: data to compare.
- ``lane_mask``: mask of the active lanes.

**Return value**

- ``true`` if all lanes in the ``lane_mask`` have the same value for ``data``. ``false`` otherwise.

**Preconditions**

- ``lane_mask`` must be non-zero.
- ``T`` shall have no padding bits, that is, ``T``'s value representation shall be identical to its object representation.

..
   - The functionality is only supported on ``SM >= 70``.

**Undefined Behavior**

- ``lane_mask`` must represent a subset of the active lanes, undefined behavior otherwise.

**Performance considerations**

- The function is slightly faster when called with a mask of all active lanes (overload function) even if all lanes participates in the call.
- The function is slower when called with a non-fully active warp.

..
   - The function calls the PTX instruction ``match.sync`` :math:`ceil\left(\frac{sizeof(data)}{4}\right)` times.

..
   **References**

   - `CUDA match_all Intrinsics <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-match-functions>`_
   - `PTX match.sync instruction <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync>`_

Example
-------

.. code:: cpp

    #include <cuda/std/array>
    #include <cuda/std/cassert>
    #include <cuda/warp>

    struct MyStruct {
        double x; // 8 bytes
        int    y; // 4 bytes
    };            // 4 bytes of padding

    __global__ void warp_match_kernel() {
        assert(cuda::device::warp_match_all(2));
        assert(cuda::device::warp_match_all(2, cuda::device::lane_mask::all()));
        assert(cuda::device::warp_match_all(MyStruct{1.0, 3})); // Undefined Behavior
        assert(!cuda::device::warp_match_all(threadIdx.x));
    }

    int main() {
        warp_match_kernel<<<1, 32>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/Eq81fTb8z>`_
