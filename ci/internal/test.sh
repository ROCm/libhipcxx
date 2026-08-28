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
#   BUILD_DIR    Scratch directory the build tree was created in.
#   INSTALL_DIR  Prefix handed to the test configure step, matching the one the
#                build was configured with.
#
# Optional environment variables:
#   PROJECT_ID              Checkout directory name (default: libhipcxx).
#   LIBHIPCXX_CONDA_ENV     Conda environment to activate first.
#   LIBHIPCXX_HIPRTC_TESTS  'false' skips the hiprtc suite (default: run it).
#   LIBHIPCXX_DETECT_ARCH   'true' resolves the GPU architecture from the
#                           runner with offload-arch instead of using
#                           AMDGPU_TARGETS.
#   AMDGPU_TARGETS          GPU architectures to test.
#   CMAKE_INSTALL_PREFIX    Install prefix, overriding INSTALL_DIR.
#   HIP_HIPCC_EXECUTABLE    Defaults to ${ROCM_PATH}/bin/hipcc.
#   CMAKE_VERSION           Version of the cmake package to install into the
#                           venv this script builds, which also supplies ctest.
#   LIBHIPCXX_LIT_VERSION   lit version for that venv.
#   ROCM_PATH

set -xeu

: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${INSTALL_DIR:?INSTALL_DIR must be set}"

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

# cmake, ctest and the lit version the tests need come from this venv.
if [ -n "${CMAKE_VERSION:-}" ]; then
  python3 --version
  python3 -m venv --clear _venv
  # shellcheck disable=SC1091
  . _venv/bin/activate
  pip3 install --upgrade pip
  # ninja too: the presets generate Ninja, and the build trees shipped in the tests
  # tarball cache CMAKE_MAKE_PROGRAM as a path into this venv, which build.sh excludes
  # from the tarball. Without it the reconfigure fails on a dangling ninja path.
  pip3 install "cmake==${CMAKE_VERSION}" "lit==${LIBHIPCXX_LIT_VERSION}" ninja psutil
fi
cmake --version && ctest --version

if [ "${LIBHIPCXX_DETECT_ARCH:-false}" == "true" ]; then
  # Match on ^gfx rather than taking the last line: a bare `| tail -n1` returns tail's
  # exit status, so a failing detector slips past `set -e` and leaves AMDGPU_TARGETS
  # empty, which silently drops the three -D flags below and builds every arch.
  # Discard gfx000 (the CPU agent) before picking a line, not after: selecting first
  # and filtering second yields nothing on a host that reports the CPU agent ahead of
  # the GPU.
  detect_gfx() { "$@" 2>/dev/null | grep -E '^gfx[0-9a-f]+' | grep -v '^gfx000$' | head -n1; }
  AMDGPU_TARGETS="$(detect_gfx offload-arch || true)"
  if [ -z "${AMDGPU_TARGETS}" ]; then
    AMDGPU_TARGETS="$(detect_gfx rocm_agent_enumerator || true)"
  fi
  if [ -z "${AMDGPU_TARGETS}" ]; then
    echo "error: LIBHIPCXX_DETECT_ARCH=true but neither offload-arch nor rocm_agent_enumerator" \
         "returned a gfx target; refusing to fall back to an all-arch build" >&2
    exit 1
  fi
  echo "detected AMDGPU_TARGETS=${AMDGPU_TARGETS}"
fi

HIP_HIPCC_EXECUTABLE="${HIP_HIPCC_EXECUTABLE:-${ROCM_PATH:-/opt/rocm}/bin/hipcc}"

# Pip-installed ROCm SDK puts libamdhip64.so under ${ROCM_PATH}/lib, which is not
# on the default loader path. c2h binaries have no RPATH, so set it here.
export LD_LIBRARY_PATH="${ROCM_PATH:-/opt/rocm}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

install_prefix="${CMAKE_INSTALL_PREFIX:-${INSTALL_DIR}}"

cmake_options=(
  "-DHIP_HIPCC_EXECUTABLE=${HIP_HIPCC_EXECUTABLE}"
  # No fetching during a test run: the build already satisfied all dependencies.
  "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
  ${AMDGPU_TARGETS:+"-DCMAKE_HIP_ARCHITECTURES=${AMDGPU_TARGETS}"}
  ${AMDGPU_TARGETS:+"-DGPU_TARGETS=${AMDGPU_TARGETS}"}
  ${AMDGPU_TARGETS:+"-DAMDGPU_TARGETS=${AMDGPU_TARGETS}"}
  "-DCMAKE_INSTALL_PREFIX=${install_prefix}"
)

# Test libhipcxx without hiprtc; writes its config into build/libcudacxx-cpp17
bash ./ci/test_libhipcxx.sh -cmake-options "${cmake_options[*]}"

# Test libhipcxx with hiprtc; writes its config into build/libcudacxx-nvrtc-cpp17
if [ "${LIBHIPCXX_HIPRTC_TESTS:-true}" == "true" ]; then
  bash ./ci/hiprtc_libhipcxx.sh -cmake-options "${cmake_options[*]}"
fi
