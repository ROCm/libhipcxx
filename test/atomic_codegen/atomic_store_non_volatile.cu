// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cuda/atomic>

__global__ void store_relaxed_device_non_volatile(int* data, int in)
{
  auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
  ref.store(in, cuda::std::memory_order_relaxed);
}

/*

; SM8X-LABEL: .target sm_80
; SM8X:      .visible .entry [[FUNCTION:_.*store_relaxed_device_non_volatile.*]](
; SM8X-DAG:  ld.param.u64 %rd[[#ATOM:]], [[[FUNCTION]]_param_0];
; SM8X-DAG:  ld.param.u32 %r[[#INPUT:]], [[[FUNCTION]]_param_1];
; SM8X-NEXT: //
; SM8X-NEXT: st.relaxed.gpu.b32 [%rd[[#ATOM]]], %r[[#INPUT]];
; SM8X-NEXT: //
; SM8X-NEXT: ret;


; ----- AMD/HIP additions below (LLVM-IR check; see test/atomic_codegen/CMakeLists.txt) -----
; HIP_IR-LABEL: define{{.*}}amdgpu_kernel void @{{.*}}store_relaxed_device_non_volatile{{.*}}
; HIP_IR:       store atomic {{(volatile )?}}i32 %{{[^,]+}}, ptr %{{[^ ]+}} syncscope("agent") monotonic
; HIP_IR:       ret void

*/
