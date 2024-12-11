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
// UNSUPPORTED: c++98, c++03

// NOTE: atomic<> of a TriviallyCopyable class is wrongly rejected by older
// clang versions. It was fixed right before the llvm 3.5 release. See PR18097.
// XFAIL: apple-clang-6.0, clang-3.4, clang-3.3

// <cuda/std/atomic>

// constexpr atomic<T>::atomic(T value)

#include <hip/std/atomic>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "cuda_space_selector.h"

struct UserType {
    int i;

    __host__ __device__
    UserType() noexcept {}
    __host__ __device__
    constexpr explicit UserType(int d) noexcept : i(d) {}

    __host__ __device__
    friend bool operator==(const UserType& x, const UserType& y) {
        return x.i == y.i;
    }
};

template <class Tp, template<typename, typename> class, hip::thread_scope Scope>
struct TestFunc {
    __host__ __device__
    void operator()() const {
        typedef hip::atomic<Tp, Scope> Atomic;
        static_assert(hip::std::is_literal_type<Atomic>::value, "");
        constexpr Tp t(42);
        {
            constexpr Atomic a(t);
            assert(a == t);
        }
        {
            constexpr Atomic a{t};
            assert(a == t);
        }
        #if !defined(_GNUC_VER) || _GNUC_VER >= 409
        // TODO: Figure out why this is failing with GCC 4.8.2 on CentOS 7 only.
        {
            constexpr Atomic a = ATOMIC_VAR_INIT(t);
            assert(a == t);
        }
        #endif
    }
};


int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
    TestFunc<UserType, local_memory_selector, hip::thread_scope_system>()();
    TestEachIntegralType<TestFunc, local_memory_selector, hip::thread_scope_system>()();
#endif
    TestFunc<UserType, local_memory_selector, hip::thread_scope_device>()();
    TestEachIntegralType<TestFunc, local_memory_selector, hip::thread_scope_device>()();
    TestFunc<UserType, local_memory_selector, hip::thread_scope_block>()();
    TestEachIntegralType<TestFunc, local_memory_selector, hip::thread_scope_block>()();

  return 0;
}
