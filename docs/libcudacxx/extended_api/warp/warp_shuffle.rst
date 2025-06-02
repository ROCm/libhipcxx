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
  :description: API reference for cuda::device warp shuffle functions, which exchange data of arbitrary size between threads within a warp in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, warp_shuffle, warp_shuffle_idx, warp_shuffle_up, warp_shuffle_down, warp_shuffle_xor

.. _libcudacxx-extended-api-warp-warp-shuffle:

Warp Shuffle
============

Defined in ``<cuda/warp>`` header.

``warp_shuffle_idx``:

.. code:: cuda

    namespace cuda::device {

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_idx(const T& data,
                     int      src_lane,
                     uint32_t lane_mask = 0xFFFFFFFF,
                     cuda::std::integral_constant<int, Width> = {})

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_idx(const T& data,
                     int      src_lane,
                     cuda::std::integral_constant<int, Width>) // lane_mask is 0xFFFFFFFF

    } // namespace cuda::device

``warp_shuffle_up``:

.. code:: cuda

    namespace cuda::device {

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_up(const T& data,
                    int      delta,
                    uint32_t lane_mask = 0xFFFFFFFF,
                    cuda::std::integral_constant<int, Width> = {})

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_up(const T& data,
                    int      delta,
                    cuda::std::integral_constant<int, Width>) // lane_mask is 0xFFFFFFFF

    } // namespace cuda::device

``warp_shuffle_down``:

.. code:: cuda

    namespace cuda::device {

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_down(const T& data,
                      int      delta,
                      uint32_t lane_mask = 0xFFFFFFFF,
                      cuda::std::integral_constant<int, Width> = {})

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_down(const T& data,
                      int      delta,
                      cuda::std::integral_constant<int, Width>) // lane_mask is 0xFFFFFFFF

    } // namespace cuda::device

``warp_shuffle_xor``:

.. code:: cuda

    namespace cuda::device {

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_xor(const T& data,
                     int      xor_mask,
                     uint32_t lane_mask = 0xFFFFFFFF,
                     cuda::std::integral_constant<int, Width> = {})

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_xor(const T& data,
                     int      xor_mask,
                     cuda::std::integral_constant<int, Width>) // lane_mask is 0xFFFFFFFF

    } // namespace cuda::device

Result type:

.. code:: cuda

    namespace cuda::device {

    template <typename T>
    struct warp_shuffle_result {
        T    data;
        bool pred;

        __device__ operator T() const { return data; }
    };

    } // namespace cuda::device

The functionality provides a generalized and safe alternative to CUDA warp shuffle intrinsics.
The functions allow to exchange data of any data size, including raw arrays, pointers, and structs.

**Parameters**

- ``data``: data to exchange.
- ``src_lane``: source lane.
- ``delta``: offset from the source lane.
- ``xor_mask``: XOR mask to apply to the source lane.

**Return value**

``warp_shuffle_result``:

- ``data``: data of the destination lane.
- ``pred``: ``true`` if the destination lane is within the source lane window. ``false`` otherwise.

**Constraints**

- ``Width`` must be a power of two in the range [1, 32]
- ``T``: only ``void`` pointers are allowed to avoid bug-prone code

**Preconditions**

- ``lane_mask`` must be a subset of the active mask
- The destination lane must be a member of the ``lane_mask``
- ``delta`` and ``xor_mask`` must be less than ``Width``. Modulo behavior is allowed for ``src_lane``
- All lanes must have the same value for ``lane_mask``, ``delta`` and ``xor_mask``

**Performance considerations**

- The function performs a 32-bit warp shuffle :math:`ceil\left(\frac{sizeof(data)}{4}\right)` times.

**References**

- `HIP warp shuffle functions <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#warp-shuffle-functions>`_

Example
-------

.. code:: cuda

    #include <cuda/std/array>
    #include <cuda/std/type_traits>
    #include <cuda/warp>
    #include <cstdio>

    struct MyStruct {
        double x;
        int    y;
    };

    __global__ void warp_shuffle_kernel() {
        cuda::std::integral_constant<int, 16> half_warp;
        auto                     laneid      = __lane_id();
        int                      raw_array[] = {threadIdx.x, threadIdx.x + 1, threadIdx.x + 2};
        cuda::std::array<int, 3> array       = {threadIdx.x, threadIdx.x + 1, threadIdx.x + 2};
        MyStruct                 my_structs{static_cast<double>(threadIdx.x), threadIdx.x + 1};
        if (laneid < 16) {
            // lanes [0, 15] get an array with values {5, 6, 7}
            auto ret = cuda::device::warp_shuffle_idx(raw_array, 5, 0xFFFF, half_warp);
            printf("lane %2d: [%d, %d, %d]\n", laneid, ret.data[0], ret.data[1], ret.data[2]);

            // lanes [1, 15] get an array with values {threadIdx.x - 1, threadIdx.x, threadIdx.x + 1}
            // lane 0 keeps the original values
            auto array_ret = cuda::device::warp_shuffle_up(array, 1, half_warp).data;
            printf("lane %2d: [%d, %d, %d]\n", laneid, array[0], array[1], array_ret[2]);
        }
        // lanes [0, 13] get my_structs with values {threadIdx.x + 2, threadIdx.x + 3} and pred=true
        auto ret = cuda::device::warp_shuffle_down<16>(my_structs, 2);
        printf("lane %2d: {%f, %d}, pred %d\n", laneid, ret.data.x, ret.data.y, ret.pred);
    }

    int main() {
        warp_shuffle_kernel<<<1, 32>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/soWTaG6Eb>`_
