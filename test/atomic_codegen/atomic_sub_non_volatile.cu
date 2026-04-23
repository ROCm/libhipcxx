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

__global__ void sub_relaxed_device_non_volatile(int* data, int* out, int n)
{
  auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
  *out     = ref.fetch_sub(n, cuda::std::memory_order_relaxed);
}

/*

; SM8X-LABEL: .target sm_80
; SM8X:      .visible .entry [[FUNCTION:_.*sub_relaxed_device_non_volatile.*]](
// <<<<<<< OLD CODE from 70d447d5db (4144c37430) - COMMENTED OUT
// ; SM8X-DAG:  ld.param.u64 %rd[[#ATOM:]], [[[FUNCTION]]_param_0];
// ; SM8X-DAG:  ld.param.u64 %rd[[#EXPECTED:]], [[[FUNCTION]]_param_1];
// ; SM8X-DAG:  ld.param.u32 %r[[#INPUT:]], [[[FUNCTION]]_param_2];
// ; SM8X-NEXT: cvta.to.global.u64 %rd[[#GOUT:]], %rd[[#EXPECTED]];
// ; SM8X-NEXT: neg.s32 %r[[#NEG:]], %r[[#INPUT]];
// ; SM8X-NEXT: //
// ; SM8X-NEXT: atom.add.relaxed.gpu.u32 %r[[#DEST:]],[%rd[[#ATOM]]],%r[[#NEG]];
// ; SM8X-NEXT: //
// ; SM8X-NEXT: st.global.u32 [%rd[[#GOUT]]], %r[[#DEST]];
// =======
; SM8X-DAG:  ld.param.{{b|u}}64 %rd[[#ATOM:]], {{.*}}[[FUNCTION]]_param_0{{.*}}
; SM8X-DAG:  ld.param.{{b|u}}64 %rd[[#RESULT:]], {{.*}}[[FUNCTION]]_param_1{{.*}}
; SM8X-DAG:  ld.param.{{b|u}}32 %r[[#INPUT:]], {{.*}}[[FUNCTION]]_param_2{{.*}}
; SM8X-DAG:  cvta.to.global.u64 %rd[[#GOUT:]], %rd[[#RESULT]];
; SM8X-NEXT: neg.s32 %r[[#NEG:]], %r[[#INPUT]];
; SM8X-NEXT: {{/*[[:space:]] *}}atom.add.relaxed.gpu.s32 %r[[#DEST:]],[%rd[[#ATOM]]],%r[[#NEG]];{{[[:space:]]/*}}
; SM8X-NEXT: st.global.{{b|u}}32 [%rd[[#GOUT]]], %r[[#DEST]];
// >>>>>>> END NEW CODE (4144c37430)
; SM8X-NEXT: ret;


; ----- AMD/HIP additions below (LLVM-IR check; see test/atomic_codegen/CMakeLists.txt) -----
; HIP_IR-LABEL: define{{.*}}amdgpu_kernel void @{{.*}}sub_relaxed_device_non_volatile{{.*}}
; HIP_IR:       atomicrmw {{(volatile )?}}add ptr %{{[^,]+}}, i32 %{{[^ ]+}} syncscope("agent") monotonic
; HIP_IR:       store i32 %{{[^,]+}}, ptr addrspace(1) %{{[^,]+}}
; HIP_IR:       ret void

*/
