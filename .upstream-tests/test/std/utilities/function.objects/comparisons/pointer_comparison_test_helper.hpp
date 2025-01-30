// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

#ifndef POINTER_COMPARISON_TEST_HELPER_HPP
#define POINTER_COMPARISON_TEST_HELPER_HPP

#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "test_macros.h"

template <class T, template<class> class CompareTemplate>
#ifdef __HIP_PLATFORM_AMD__
_LIBCUDACXX_INLINE_VISIBILITY
#endif
void do_pointer_comparison_test() {
    typedef CompareTemplate<T*> Compare;
    typedef CompareTemplate<std::uintptr_t> UIntCompare;
#if TEST_STD_VER > 11
    typedef CompareTemplate<void> VoidCompare;
#else
    typedef Compare VoidCompare;
#endif
    std::vector<std::shared_ptr<T> > pointers;
    const std::size_t test_size = 100;
    for (size_t i=0; i < test_size; ++i)
        pointers.push_back(std::shared_ptr<T>(new T()));
    Compare comp;
    UIntCompare ucomp;
    VoidCompare vcomp;
    for (size_t i=0; i < test_size; ++i) {
        for (size_t j=0; j < test_size; ++j) {
            T* lhs = pointers[i].get();
            T* rhs = pointers[j].get();
            std::uintptr_t lhs_uint = reinterpret_cast<std::uintptr_t>(lhs);
            std::uintptr_t rhs_uint = reinterpret_cast<std::uintptr_t>(rhs);
            assert(comp(lhs, rhs) == ucomp(lhs_uint, rhs_uint));
            assert(vcomp(lhs, rhs) == ucomp(lhs_uint, rhs_uint));
        }
    }
}

#endif // POINTER_COMPARISON_TEST_HELPER_HPP
