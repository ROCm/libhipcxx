#!/usr/bin/env bash

# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -euo pipefail

# HIP analog of atomic_codegen/dump_and_check.bash for the simd codegen tests.
#
# Usage:
#   dump_and_check_hip.bash <test.cu> <prefix>
#
# Compiles the simd codegen test as HIP device-only to AMDGCN assembly (clang -S)
# and FileChecks the ISA against the <prefix> (HIP_ISA) lines embedded in the
# test's trailing comment -- i.e. asserts that cuda::std::simd operators lower to
# the packed vector instructions (v_pk_add_f16 / v_pk_mul_f16 / v_pk_fma_f16 /
# v_pk_add_f32 / ...), the AMDGCN equivalent of the NVIDIA HADD2/HFMA2 SASS check.
#
# Honours:
#   $FILECHECK              (default: bare 'FileCheck' via $PATH; CMake forwards
#                            the absolute path find_program() located)
#   $LIBCUDACXX_SOURCE_DIR  (required: repo root, for -I include + force_include)
#   $HIP_ISA_ARCH           (default gfx90a)
#   $HIP_ISA_CXX            (default /opt/rocm/lib/llvm/bin/clang++)
#
# FileCheck runs strict, so a .cu with no HIP_ISA check lines fails the build.

input_testfile="$1"
input_prefix="$2"

filecheck="${FILECHECK:-FileCheck}"
LH="${LIBCUDACXX_SOURCE_DIR:?LIBCUDACXX_SOURCE_DIR must be set on the HIP path}"
arch="${HIP_ISA_ARCH:-gfx90a}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

asm_out="$(mktemp /tmp/simd_codegen_hip.XXXXXX.s)"
trap 'rm -f "$asm_out"' EXIT

"${HIP_ISA_CXX:-/opt/rocm/lib/llvm/bin/clang++}" \
    -O3 -x hip --offload-arch="${arch}" -S --cuda-device-only \
    -std=c++17 -fno-rtti -Wno-comment \
    -include "$LH/test/libcudacxx/force_include_hip.h" \
    -I "$here/hip_compat" \
    -I "$LH/include" -I "$LH/test/support" -I /opt/rocm/include \
    -D_CCCL_NO_SYSTEM_HEADER \
    "$input_testfile" -o "$asm_out"

"$filecheck" --check-prefix "$input_prefix" "$input_testfile" < "$asm_out"
