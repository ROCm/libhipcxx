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

# Run the libhipcxx test suites against a tree produced by ci/internal/build.sh,
# by driving the project's own ci/test_libhipcxx.sh and ci/hiprtc_libhipcxx.sh.
#
# The tester images reach this script through the source tree shipped inside
# the tests tarball.
#
# Required environment variables:
#   BUILD_DIR  Scratch directory the build tree was created in.
#
# Optional environment variables:
#   PROJECT_ID              Checkout directory name (default: libhipcxx).
#   LIBHIPCXX_CONDA_ENV     Conda environment to activate first.
#   LIBHIPCXX_HIPRTC_TESTS  'false' skips the hiprtc suite (default: run it).
#   LIBHIPCXX_DETECT_ARCH   'true' resolves the GPU architecture from the
#                           runner with offload-arch instead of using
#                           AMDGPU_TARGETS.
#   AMDGPU_TARGETS          GPU architectures to test.
#   CMAKE_INSTALL_PREFIX    Install prefix handed to the test configure step.
#   HIP_HIPCC_EXECUTABLE    Defaults to ${ROCM_PATH}/bin/hipcc.
#   CMAKE_VERSION           Only used to note that build.sh created a venv.
#   ROCM_PATH

set -xeu

: "${BUILD_DIR:?BUILD_DIR must be set}"

PROJECT_ID="${PROJECT_ID:-libhipcxx}"
build_src_dir="${BUILD_DIR}/${PROJECT_ID}"

if [ -n "${LIBHIPCXX_CONDA_ENV:-}" ]; then
  set +u # note: conda script may have unbound variables
  source ${CONDA_DIR}/etc/profile.d/conda.sh
  conda activate ${LIBHIPCXX_CONDA_ENV}
  set -u
fi

# Keep the host toolchain build.sh used. CCC_OVERRIDE_OPTIONS is read by the
# clang driver per invocation and is not persisted in the CMake cache, so
# without it the test relink falls back to the platform libstdc++.
if [ -d "/opt/rh/gcc-toolset-$(g++ -dumpversion)" ]; then
  toolchain="/opt/rh/gcc-toolset-$(g++ -dumpversion)/root/usr"
  export CCC_OVERRIDE_OPTIONS="+--gcc-toolchain=${toolchain}"
  # note: in some shells, cmake is using /bin/{cc, c++} as compilers,
  # which may not be the compilers of the enabled toolchain.
  export CC=${toolchain}/bin/cc
fi

# NOTE: this only takes effect when the tree has not been configured yet -- CMake
# reads $CXX on the first configure and the cache wins on every later one, so a
# tree built by ci/internal/build.sh keeps that script's host compiler
# (amdclang++) no matter what is set here. Kept for a from-scratch test run.
export CXX="${CXX_FOR_TESTS:-hipcc}"

# NOTE: We must execute the tests from the build tree because they need the
# build artifact <build>/test/lit.site.cfg
cd "${build_src_dir}"

# build.sh only creates this venv when the caller pins CMAKE_VERSION; it holds
# cmake and the lit version the tests need.
if [ -f _venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . _venv/bin/activate
  pip3 install psutil
fi

if [ "${LIBHIPCXX_DETECT_ARCH:-false}" == "true" ]; then
  AMDGPU_TARGETS="$(offload-arch | tail -n1)"
fi

HIP_HIPCC_EXECUTABLE="${HIP_HIPCC_EXECUTABLE:-${ROCM_PATH:-/opt/rocm}/bin/hipcc}"

cmake_options=(
  "-DHIP_HIPCC_EXECUTABLE=${HIP_HIPCC_EXECUTABLE}"
  # No fetching during a test run: the build already satisfied all dependencies.
  "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
  ${AMDGPU_TARGETS:+"-DCMAKE_HIP_ARCHITECTURES=${AMDGPU_TARGETS}"}
  ${AMDGPU_TARGETS:+"-DGPU_TARGETS=${AMDGPU_TARGETS}"}
  ${AMDGPU_TARGETS:+"-DAMDGPU_TARGETS=${AMDGPU_TARGETS}"}
  ${CMAKE_INSTALL_PREFIX:+"-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"}
)

# Test libhipcxx without hiprtc; writes its config into build/libcudacxx-cpp17
bash ./ci/test_libhipcxx.sh -cmake-options "${cmake_options[*]}"

# Test libhipcxx with hiprtc; writes its config into build/libcudacxx-nvrtc-cpp17
if [ "${LIBHIPCXX_HIPRTC_TESTS:-true}" == "true" ]; then
  bash ./ci/hiprtc_libhipcxx.sh -cmake-options "${cmake_options[*]}"
fi
