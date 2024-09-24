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

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC
// UNSUPPORTED: nvrtc

// <cuda/std/complex>

// Test that UDT's convertible to an integral or floating point type do not
// participate in overload resolution.

#include <hip/std/complex>
#include <hip/std/type_traits>
#include <hip/std/cassert>

template <class IntT>
struct UDT {
  operator IntT() const { return 1; }
};
#if defined(__HIP_PLATFORM_AMD__)
__host__ __device__ UDT<float> ft;
__host__ __device__ UDT<double> dt;
// CUDA treats long double as double
__host__ __device__ UDT<long double> ldt;
__host__ __device__ UDT<int> it;
__host__ __device__ UDT<unsigned long> uit;
#else
UDT<float> ft;
UDT<double> dt;
// CUDA treats long double as double
// UDT<long double> ldt;
UDT<int> it;
UDT<unsigned long> uit;
#endif

int main(int, char**)
{
    {
        hip::std::real(ft); // expected-error {{no matching function}}
        hip::std::real(dt); // expected-error {{no matching function}}
        #if defined(__HIP_PLATFORM_AMD__)
        hip::std::real(ldt); // expected-error {{no matching function}}
        #endif
        hip::std::real(it); // expected-error {{no matching function}}
        hip::std::real(uit); // expected-error {{no matching function}}
    }
    {
        hip::std::imag(ft); // expected-error {{no matching function}}
        hip::std::imag(dt); // expected-error {{no matching function}}
        #if defined(__HIP_PLATFORM_AMD__)
        hip::std::imag(ldt); // expected-error {{no matching function}}
        #endif
        hip::std::imag(it); // expected-error {{no matching function}}
        hip::std::imag(uit); // expected-error {{no matching function}}
    }
    {
        hip::std::arg(ft); // expected-error {{no matching function}}
        hip::std::arg(dt); // expected-error {{no matching function}}
        #if defined(__HIP_PLATFORM_AMD__)
        hip::std::arg(ldt); // expected-error {{no matching function}}
        #endif
        hip::std::arg(it); // expected-error {{no matching function}}
        hip::std::arg(uit); // expected-error {{no matching function}}
    }
    {
        hip::std::norm(ft); // expected-error {{no matching function}}
        hip::std::norm(dt); // expected-error {{no matching function}}
        #if defined(__HIP_PLATFORM_AMD__)
        hip::std::norm(ldt); // expected-error {{no matching function}}
        #endif
        hip::std::norm(it); // expected-error {{no matching function}}
        hip::std::norm(uit); // expected-error {{no matching function}}
    }
    {
        hip::std::conj(ft); // expected-error {{no matching function}}
        hip::std::conj(dt); // expected-error {{no matching function}}
        #if defined(__HIP_PLATFORM_AMD__)
        hip::std::conj(ldt); // expected-error {{no matching function}}
        #endif
        hip::std::conj(it); // expected-error {{no matching function}}
        hip::std::conj(uit); // expected-error {{no matching function}}
    }
    {
        hip::std::proj(ft); // expected-error {{no matching function}}
        hip::std::proj(dt); // expected-error {{no matching function}}
        #if defined(__HIP_PLATFORM_AMD__)
        hip::std::proj(ldt); // expected-error {{no matching function}}
        #endif
        hip::std::proj(it); // expected-error {{no matching function}}
        hip::std::proj(uit); // expected-error {{no matching function}}
    }

  return 0;
}
