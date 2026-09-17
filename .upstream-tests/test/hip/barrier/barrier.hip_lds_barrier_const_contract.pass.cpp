// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// UNSUPPORTED: nvcc, nvhpc, nvc++

// <cuda/barrier>
// Verify __lds_barrier_t helper member signatures and its const LDS-word pointer accessor.

#include <cuda/barrier>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
using lds_barrier = hip::__lds_barrier_t;
using const_lds_word_ptr = const hip::std::uint64_t __attribute__((address_space(3)))*;
using const_lds_word_ptr_signature = const_lds_word_ptr (lds_barrier::*)() const;
using arrive_rtn_signature = hip::std::uint64_t (lds_barrier::*)(hip::std::uint32_t);
using async_arrive_signature = void (lds_barrier::*)();

static_assert(hip::std::is_same<decltype(static_cast<const_lds_word_ptr_signature>(&lds_barrier::__lds_word_ptr)),
                                const_lds_word_ptr_signature>::value,
              "__lds_word_ptr() const must return a const LDS word pointer");
static_assert(hip::std::is_same<decltype(static_cast<arrive_rtn_signature>(&lds_barrier::__arrive_rtn)),
                                arrive_rtn_signature>::value,
              "__arrive_rtn(uint32_t) must be non-const");
static_assert(hip::std::is_same<decltype(static_cast<async_arrive_signature>(&lds_barrier::__async_arrive)),
                                async_arrive_signature>::value,
              "__async_arrive() must be non-const");
#endif // _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT

int main(int, char**)
{
  return 0;
}