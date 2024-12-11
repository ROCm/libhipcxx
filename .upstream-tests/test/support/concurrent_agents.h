//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#if !(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
    #include <thread>
#endif

template<typename... Fs>
__host__ __device__
void concurrent_agents_launch(Fs ...fs)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#if !defined(__HIP_PLATFORM_AMD__)
    #if __CUDA_ARCH__ < 350
        #error "This test requires CUDA dynamic parallelism to work."
    #endif
#endif
    assert(blockDim.x == sizeof...(Fs));

    using fptr = void (*)(void *);

    fptr device_threads[] = {
        [](void * data) {
            (*reinterpret_cast<Fs *>(data))();
        }...
    };

    void * device_thread_data[] = {
        reinterpret_cast<void *>(&fs)...
    };

    __syncthreads();

    device_threads[threadIdx.x](device_thread_data[threadIdx.x]);

    __syncthreads();

#else

    std::thread threads[]{
        std::thread{ std::forward<Fs>(fs) }...
    };

    for (auto && thread : threads)
    {
        thread.join();
    }

#endif
}

