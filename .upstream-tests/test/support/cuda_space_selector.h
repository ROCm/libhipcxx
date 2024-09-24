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

#include <hip/std/cstddef>

#if defined(__CUDACC__) && !defined(__NVCOMPILER)
#define __exec_check_disable__ #pragma nv_exec_check_disable
#else
#define __exec_check_disable__
#endif

#if defined(__CUDACC_RTC__) || !defined(__HIP_DEVICE_COMPILE__)
#define LAMBDA [=]
#else
#define LAMBDA [=] __host__ __device__
#endif

#if defined (__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define SHARED __shared__
#else
#define SHARED
#endif

template<typename T, hip::std::size_t SharedOffset>
struct malloc_memory_provider {
    static const constexpr hip::std::size_t prefix_size
        = SharedOffset % alignof(T *) == 0
            ? SharedOffset
            : SharedOffset + (alignof(T *) - SharedOffset % alignof(T *));
    static const constexpr hip::std::size_t shared_offset = prefix_size + sizeof(T *);

private:
    __host__ __device__
    T *& get_pointer() {
        alignas(T*)
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        __shared__
#else
        static
#endif
        char storage[shared_offset];
        T *& allocated = *reinterpret_cast<T **>(storage + prefix_size);
        return allocated;
    }

public:
    __host__ __device__
    T * get() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (threadIdx.x == 0) {
#endif
        get_pointer() = reinterpret_cast<T *>(malloc(sizeof(T) + alignof(T)));
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        }
        __syncthreads();
#endif
        auto ptr = reinterpret_cast<hip::std::uintptr_t>(get_pointer());
        ptr += alignof(T) - ptr % alignof(T);
        return reinterpret_cast<T *>(ptr);
    }

    __host__ __device__
    ~malloc_memory_provider() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (threadIdx.x == 0) {
#endif
        free((void*)get_pointer());
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        }
        __syncthreads();
#endif
    }
};

template<typename T, hip::std::size_t SharedOffset>
struct local_memory_provider {
    static const constexpr hip::std::size_t prefix_size = 0;
    static const constexpr hip::std::size_t shared_offset = SharedOffset;

    alignas(T) char buffer[sizeof(T)];

    __host__ __device__
    T * get() {
// FIXME (HIP): A workaround for atomic ops on 8-byte types 
// (e.g., long, ulong, long long, ulonglong) allocated on local memory.
// We allocated them on shared memory for now.
// It should be removed when a fix is available.
// This issue is tracked in SWDEV-388507.
#ifdef __HIP_DEVICE_COMPILE__ 
        if(sizeof(T) > 4){
            alignas(T) char __shared__ buffer[sizeof(T)];
            return reinterpret_cast<T *>(&buffer);
        }
#endif
        return reinterpret_cast<T *>(&buffer);
    }
};

template<typename T, hip::std::size_t SharedOffset>
struct device_shared_memory_provider {
    static const constexpr hip::std::size_t prefix_size
        = SharedOffset % alignof(T) == 0
            ? SharedOffset
            : SharedOffset + (alignof(T) - SharedOffset % alignof(T));
    static const constexpr hip::std::size_t shared_offset = prefix_size + sizeof(T);

    __device__
    T * get() {
        alignas(T) __shared__ char buffer[shared_offset];
        return reinterpret_cast<T *>(buffer + prefix_size);
    }
};

struct init_initializer {
    template<typename T, typename ...Ts>
    __host__ __device__
    static void construct(T & t, Ts && ...ts) {
        t.init(std::forward<Ts>(ts)...);
    }
};

struct constructor_initializer {
    template<typename T, typename ...Ts>
    __host__ __device__
    static void construct(T & t, Ts && ...ts) {
        new ((void*)&t) T(std::forward<Ts>(ts)...);
    }
};

struct default_initializer {
    template<typename T>
    __host__ __device__
    static void construct(T & t) {
        new ((void*)&t) T;
    }
};

template<typename T,
    template<typename, hip::std::size_t> typename Provider,
    typename Initializer = constructor_initializer,
    hip::std::size_t SharedOffset = 0>
class memory_selector
{
    Provider<T, SharedOffset> provider;
    T * ptr;

public:
    template<hip::std::size_t SharedOffset_>
    using offsetted = memory_selector<T, Provider, Initializer, SharedOffset + SharedOffset_>;

    static const constexpr hip::std::size_t prefix_size = Provider<T, SharedOffset>::prefix_size;
    static const constexpr hip::std::size_t shared_offset = Provider<T, SharedOffset>::shared_offset;

#ifndef __CUDACC_RTC__
    __exec_check_disable__
#endif
    template<typename ...Ts>
    __host__ __device__
    T * construct(Ts && ...ts) {
        ptr = provider.get();
        assert(ptr);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (threadIdx.x == 0) {
#endif
        Initializer::construct(*ptr, std::forward<Ts>(ts)...);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        }
        __syncthreads();
#endif
        return ptr;
    }

#ifndef __CUDACC_RTC__
    __exec_check_disable__
#endif
    __host__ __device__
    ~memory_selector() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (threadIdx.x == 0) {
#endif
        ptr->~T();
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        }
        __syncthreads();
#endif
    }
};

template<typename T, typename Initializer = constructor_initializer>
using local_memory_selector = memory_selector<T, local_memory_provider, Initializer>;

template<typename T, typename Initializer = constructor_initializer>
using shared_memory_selector = memory_selector<T, device_shared_memory_provider, Initializer>;

template<typename T, typename Initializer = constructor_initializer>
using global_memory_selector = memory_selector<T, malloc_memory_provider, Initializer>;
