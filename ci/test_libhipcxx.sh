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

cd "$(dirname "${BASH_SOURCE[0]}")"
source "./build_common.sh"

print_environment_details

PRESET="libcudacxx-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_preset libcudacxx "$PRESET" "$CMAKE_OPTIONS"

# Build the c2h test executables before running CTest. The c2h
# tests are CMake 'add_test(NAME ... COMMAND <exe>)' registrations
# that ctest does NOT build automatically -- if this script is
# called standalone (without a prior './build_libhipcxx.sh' run)
# the c2h binaries would be missing and ctest would report them
# as "Not Run". Limit the build to the c2h umbrella target so we
# don't recompile the entire lit suite here -- the
# 'libcudacxx.test.lit.precompile' target is invoked by
# build_libhipcxx.sh on the CI build phase and re-running it here
# would compile the 3223 lit tests TWICE (once in precompile,
# once in the lit ctest entry below). When a prior
# build_libhipcxx.sh has already produced the c2h binaries this
# is a near-zero-cost ninja no-op.
if ! $CONFIGURE_ONLY; then
    pushd .. > /dev/null
    run_command "🏗️  Build libcudacxx (c2h only)" \
        cmake --build "${BUILD_DIR}/${PRESET}" \
              --target libcudacxx.test.c2h_all

    # Build the atomic codegen tests when FileCheck was found at
    # configure time. The 'libcudacxx.test.atomics.ptx' umbrella
    # target depends on per-.cu custom targets that each compile
    # the .cu (HIP path: clang -emit-llvm; NV path: cuobjdump
    # --dump-ptx on a static lib) and FileCheck the result against
    # the trailing '/* ... */' comment block in the .cu.
    #
    # If FileCheck is NOT available, 'test/atomic_codegen/CMakeLists.txt'
    # early-returns at configure time, the per-.cu targets are not
    # created, the umbrella is empty, and 'cmake --build --target
    # libcudacxx.test.atomics.ptx' would silently no-op. The check
    # below makes the skip explicit so CI logs do not show a misleading
    # "atomic codegen build OK" group when in fact nothing was built.
    #
    # The signal we read is the 'filecheck:FILEPATH=...' cache entry
    # that 'find_program(filecheck "FileCheck" HINTS ...)' populates --
    # this is the same source of truth CMake itself used at configure
    # time, so we cannot disagree.
    filecheck_path=$(grep '^filecheck:FILEPATH=' "${BUILD_DIR}/${PRESET}/CMakeCache.txt" 2>/dev/null \
                     | cut -d= -f2)
    if [ -n "$filecheck_path" ] && [ "$filecheck_path" != "filecheck-NOTFOUND" ]; then
        run_command "🏗️  Build libcudacxx (atomic codegen)" \
            cmake --build "${BUILD_DIR}/${PRESET}" \
                  --target libcudacxx.test.atomics.ptx

        # Same for the simd codegen tests (test/simd_codegen): the
        # 'libcudacxx.test.simd.sass' umbrella uses the identical POST_BUILD
        # FileCheck mechanism (HIP path: clang -S AMDGCN ISA + FileCheck the
        # v_pk_* / s_xor_b32 packed-instruction lines). It is a separate umbrella
        # target from atomics.ptx, so it must be built explicitly here too, and
        # is gated on the same FileCheck-found signal.
        run_command "🏗️  Build libcudacxx (simd codegen)" \
            cmake --build "${BUILD_DIR}/${PRESET}" \
                  --target libcudacxx.test.simd.sass
    else
        echo "atomic + simd codegen tests skipped (FileCheck not found at configure time)"
    fi

    popd > /dev/null
fi

# The libcudacxx tests are split into two presets, one for
# regular ctest tests and another that invokes the lit tests
# harness with extra options for verbosity, etc:
CTEST_PRESET="libcudacxx-ctest-cpp${CXX_STANDARD}"
LIT_PRESET="libcudacxx-lit-cpp${CXX_STANDARD}"

test_preset "libcudacxx (CTest)" ${CTEST_PRESET}

source "./sccache_stats.sh" "start"
test_preset "libcudacxx (lit)" ${LIT_PRESET}
source "./sccache_stats.sh" "end"

print_time_summary
