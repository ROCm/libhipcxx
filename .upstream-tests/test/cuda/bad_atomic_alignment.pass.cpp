//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// UNSUPPORTED: hipcc
// We made the test unsupported because hip builtin atomics 
// fail with underaligned types (SWDEV-393058) 
// As a workaround, you need to use alignas(8) 
// to align the user-defined types manually. 

// <cuda/atomic>

// hip::atomic<key>

// Original test issue:
// https://github.com/NVIDIA/libcudacxx/issues/160

#include <hip/atomic>
#include "cuda_space_selector.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

template <template<typename, typename> typename Selector>
struct TestFn {
  __host__ __device__
  void operator()() const {
    {
        struct key {
          int32_t a;
          int32_t b;
        };
        typedef hip::std::atomic<key> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        hip::std::atomic_init(&t, key{1,2});
        auto r = t.load();
        t.store(r);
        (void)t.exchange(r);
    }
    {
        struct alignas(8) key {
          int32_t a;
          int32_t b;
        };
        typedef hip::std::atomic<key> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        hip::std::atomic_init(&t, key{1,2});
        auto r = t.load();
        t.store(r);
        (void)t.exchange(r);
    }
  }
};

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    TestFn<local_memory_selector>()();
#endif
#ifdef __CUDA_ARCH__
    TestFn<shared_memory_selector>()();
    TestFn<global_memory_selector>()();
#endif

  return 0;
}
