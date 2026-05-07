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
