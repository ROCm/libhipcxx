#!/bin/bash

# MIT License
#
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

set -e

# Usage:
#   dump_and_check.bash <input> <test.cu> <prefix> [PLATFORM]
#
# PLATFORM=NV  (default): <input> is a static archive built by CMake;
#                         cuobjdump --dump-ptx + FileCheck against PTX.
# PLATFORM=HIP          : <input> is the .cu source itself; the script
#                         compiles it with clang -emit-llvm using a
#                         baked-in flag set, then FileChecks the
#                         resulting LLVM-IR against the test file.
#                         Honours $HIP_IR_CXX (default
#                         /opt/rocm/lib/llvm/bin/clang++) and
#                         $HIP_IR_ARCH (default gfx90a). FileCheck
#                         runs strict (no --allow-empty), so a .cu
#                         with no HIP_IR check lines fails the build.

input="$1"
input_testfile="$2"
input_prefix="$3"
platform="${4:-NV}"

case "$platform" in
  NV)
    cuobjdump --dump-ptx "$input" |
      FileCheck --check-prefix "$input_prefix" "$input_testfile"
    ;;
  HIP)
    LH="${LIBCUDACXX_SOURCE_DIR:?LIBCUDACXX_SOURCE_DIR must be set on the HIP path}"
    ll_out="$(mktemp /tmp/atomic_codegen_hip.XXXXXX.ll)"
    trap 'rm -f "$ll_out"' EXIT
    "${HIP_IR_CXX:-/opt/rocm/lib/llvm/bin/clang++}" \
        -O3 -x hip --offload-arch="${HIP_IR_ARCH:-gfx90a}" \
        -S -emit-llvm --cuda-device-only \
        -std=c++17 -fno-rtti \
        -include "$LH/test/libcudacxx/force_include_hip.h" \
        -I "$LH/include" -I "$LH/test/support" -I /opt/rocm/include \
        -D_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE=1 \
        -D_CCCL_NO_SYSTEM_HEADER \
        -DCCCL_ENABLE_OPTIONAL_REF \
        -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT \
        -DLIBCUDACXX_IGNORE_DEPRECATED_ABI \
        "$input" -o "$ll_out"
    FileCheck --check-prefix "$input_prefix" "$input_testfile" < "$ll_out"
    ;;
  *)
    echo "dump_and_check.bash: unknown platform '$platform' (NV or HIP)" >&2
    exit 2
    ;;
esac
