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

# Build libhipcxx for CI: drives the project's own ci/build_libhipcxx.sh, then
# produces the CPack packages and the tests tarball the tester images consume.
#
# Both callers go through this script: the Jenkins pipeline on GPU-capable
# hosts, and the platform CI Docker builders on CPU-only hosts. Both use the
# same host compiler (amdclang++, see LIBHIPCXX_HOST_COMPILER below), so the
# two host types differ only in whether a GPU is present at build time.
#
# Required environment variables:
#   SRC_DIR              Parent directory of the project checkout.
#   BUILD_DIR            Scratch directory for the build tree.
#   INSTALL_DIR          Prefix the install step targets and the packages are
#                        built for.
#   BUILD_ARTIFACTS_DIR  Destination for packages and the tests tarball.
#
# Optional environment variables:
#   PROJECT_ID                Checkout directory name under SRC_DIR
#                             (default: libhipcxx).
#   LIBHIPCXX_BUILD_TYPE      build | install (default: build).
#   LIBHIPCXX_CONDA_ENV       Conda environment to activate first.
#   LIBHIPCXX_HOST_COMPILER   Host compiler handed to ci/build_libhipcxx.sh.
#                             Defaults to amdclang++ from the ROCm install,
#                             which works on both host types. Neither
#                             alternative does: hipcc compiles even a plain
#                             .cpp as HIP and so needs an offload arch it
#                             cannot detect on a GPU-less builder, while a
#                             distro g++ chokes on the clang-only flags
#                             (-x hip, --offload-arch) that hip::device puts
#                             on plain CXX targets.
#   LIBHIPCXX_CXX_STANDARD    C++ standard selecting the preset
#                             (default: 17; keep in sync with ci/build_common.sh).
#   LIBHIPCXX_CPACK_GENERATORS  Semicolon separated CPack generator list.
#                             Defaults to the generators whose backend is
#                             present (RPM and/or DEB).
#   LIBHIPCXX_LIT_VERSION     lit version installed into the venv.
#   CMAKE_INSTALL_PREFIX      Install prefix, overriding INSTALL_DIR; the
#                             literal '<conda-prefix>' resolves to CONDA_PREFIX.
#   CPACK_PACKAGING_INSTALL_PREFIX  Prefix baked into the packages. Defaults to
#                             the install prefix.
#   AMDGPU_TARGETS            GPU architectures to compile for. Always passed
#                             explicitly so nothing tries to detect a local GPU.
#   HIP_HIPCC_EXECUTABLE      Defaults to ${ROCM_PATH}/bin/hipcc. Required with
#                             a TheRock ROCm installation.
#   CMAKE_VERSION             Version of the cmake package to install into the
#                             venv that supplies cmake, cpack, ctest and the
#                             other python build tools. Unset means whatever is
#                             on PATH, e.g. from a conda environment.
#   ROCM_PATH, CMAKE_PREFIX_PATH
#   CMAKE_BUILD_PARALLEL_LEVEL, MAX_JOBS   Build and lit parallelism.

set -xeu

: "${SRC_DIR:?SRC_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${INSTALL_DIR:?INSTALL_DIR must be set}"
: "${BUILD_ARTIFACTS_DIR:?BUILD_ARTIFACTS_DIR must be set}"

PROJECT_ID="${PROJECT_ID:-libhipcxx}"

[[ "${LIBHIPCXX_BUILD_TYPE:-build}" == "install" ]] && do_install="1"

# note: keep in sync with ci/build_common.sh
CXX_STANDARD="${LIBHIPCXX_CXX_STANDARD:-17}"
PRESET="libcudacxx-cpp${CXX_STANDARD}"

tests_tarball_name="${PROJECT_ID}-tests.tar.gz"

if [ -n "${LIBHIPCXX_CONDA_ENV:-}" ]; then
  set +u # note: conda script may have unbound variables
  source ${CONDA_DIR}/etc/profile.d/conda.sh
  conda activate ${LIBHIPCXX_CONDA_ENV}
  set -u
fi

[[ "${CMAKE_INSTALL_PREFIX:-}" == "<conda-prefix>" ]] && install_to_conda_prefix="1"

if [ -d "/opt/rh/gcc-toolset-$(g++ -dumpversion)" ]; then
  toolchain="/opt/rh/gcc-toolset-$(g++ -dumpversion)/root/usr"
  export CCC_OVERRIDE_OPTIONS="+--gcc-toolchain=${toolchain}"
  # note: in some shells, cmake is using /bin/{cc, c++} as compilers,
  # which may not be the compilers of the enabled toolchain.
  export CC=${toolchain}/bin/cc
fi

# ci/build_common.sh takes the host compiler from CXX. Resolve amdclang++ out of
# the ROCm install rather than trusting PATH, and fail loudly if it is missing:
# falling back to hipcc or g++ would resurface the failures described above as
# confusing compile errors deep in a dependency.
if [ -z "${LIBHIPCXX_HOST_COMPILER:-}" ]; then
  rocm_root="${ROCM_PATH:-}"
  if [ -z "${rocm_root}" ] && command -v rocm-sdk >/dev/null 2>&1; then
    # ROCm >= 7.14 ships as a pip SDK with no /opt/rocm.
    rocm_root="$(rocm-sdk path --root)"
  fi
  rocm_root="${rocm_root:-/opt/rocm}"

  if [ -x "${rocm_root}/bin/amdclang++" ]; then
    LIBHIPCXX_HOST_COMPILER="${rocm_root}/bin/amdclang++"
  elif command -v amdclang++ >/dev/null 2>&1; then
    LIBHIPCXX_HOST_COMPILER="$(command -v amdclang++)"
  else
    echo "ERROR: amdclang++ not found in '${rocm_root}/bin' or on PATH." >&2
    echo "       Set LIBHIPCXX_HOST_COMPILER to override the host compiler." >&2
    exit 2
  fi
fi
export CXX="${LIBHIPCXX_HOST_COMPILER}"

#### packaging backends

if [ -n "${LIBHIPCXX_CPACK_GENERATORS:-}" ]; then
  cpack_generators="${LIBHIPCXX_CPACK_GENERATORS}"
else
  detected=()
  if command -v rpmbuild >/dev/null 2>&1; then
    detected+=("RPM")
  fi
  if command -v dpkg-deb >/dev/null 2>&1; then
    detected+=("DEB")
  fi
  cpack_generators="$(IFS=';'; echo "${detected[*]:-}")"
fi

if [ -z "${cpack_generators}" ]; then
  echo "ERROR: no CPack backend available (need rpmbuild and/or dpkg-deb)" >&2
  exit 2
fi

#### source tree

src_dir=${SRC_DIR}/${PROJECT_ID}
build_src_dir=${BUILD_DIR}/${PROJECT_ID}
# ci/build_common.sh builds into <source>/build/<preset>
build_build_dir=${build_src_dir}/build/${PRESET}

rm -rf ${build_src_dir}
mkdir -p ${BUILD_DIR}
cp -R ${src_dir} ${build_src_dir}

mkdir -p ${BUILD_ARTIFACTS_DIR}

cd ${build_src_dir}

# cmake, cpack, ctest and the test tooling come from this venv.
if [ -n "${CMAKE_VERSION:-}" ]; then
  python3 --version
  python3 -m venv _venv # note: a venv keeps this out of the conda env
  . _venv/bin/activate
  pip3 install --upgrade pip
  pip3 install "cmake==${CMAKE_VERSION}"
  pip3 install "lit==${LIBHIPCXX_LIT_VERSION}" # specific requirement for libhipcxx testing
  pip3 install ninja
  pip3 install sccache
  pip3 install psutil
fi
cmake --version && cpack --version && ctest --version

export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}${ROCM_PATH:+:${ROCM_PATH}/lib/cmake}"

# NOTE: -DHIP_HIPCC_EXECUTABLE required with TheRock ROCm installation
# according to https://github.com/ROCm/TheRock/pull/2018; see file:
# build_tools/github_actions/test_executable_scripts/test_libhipcxx.py
HIP_HIPCC_EXECUTABLE="${HIP_HIPCC_EXECUTABLE:-${ROCM_PATH:-/opt/rocm}/bin/hipcc}"

#### configure and build

if [ -n "${install_to_conda_prefix:-}" ]; then
  resolved_install_prefix="${CONDA_PREFIX}"
else
  resolved_install_prefix="${CMAKE_INSTALL_PREFIX:-${INSTALL_DIR}}"
fi
package_prefix="${CPACK_PACKAGING_INSTALL_PREFIX:-${resolved_install_prefix}}"

cmake_options=(
  "-DHIP_HIPCC_EXECUTABLE=${HIP_HIPCC_EXECUTABLE}"
  # rocm-cmake's rocm_create_package() otherwise puts a hard rocm-core
  # requirement on the generated RPM/DEB. A pip/TheRock ROCm install ships no
  # rocm-core package, so that requirement is unsatisfiable wherever these
  # packages are consumed.
  "-DROCM_DEP_ROCMCORE=OFF"
  "-DCPACK_OUTPUT_FILE_PREFIX=${BUILD_ARTIFACTS_DIR}"
  "-DCPACK_GENERATOR=${cpack_generators}"
  "-Dlibcudacxx_LIT_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS:-1}}"
  ${AMDGPU_TARGETS:+"-DCMAKE_HIP_ARCHITECTURES=${AMDGPU_TARGETS}"}
  ${AMDGPU_TARGETS:+"-DGPU_TARGETS=${AMDGPU_TARGETS}"}
  ${AMDGPU_TARGETS:+"-DAMDGPU_TARGETS=${AMDGPU_TARGETS}"}
  "-DCMAKE_INSTALL_PREFIX=${resolved_install_prefix}"
  "-DCPACK_PACKAGING_INSTALL_PREFIX=${package_prefix}"
)

bash ./ci/build_libhipcxx.sh -cmake-options "${cmake_options[*]}"

# NOTE(HIP/AMD): the hiprtc preset needs its own build tree, configured and packed
# into the tests tarball here -- the test stage cannot produce one, so without this
# the hiprtc suite fails. Configure only; hiprtcc compiles the tests at runtime.
if [ "${LIBHIPCXX_HIPRTC_TESTS:-true}" == "true" ]; then
  bash ./ci/hiprtc_libhipcxx.sh -configure -cmake-options "${cmake_options[*]}"
fi

#### package and install

cmake --build ${build_build_dir} --target package

if [ -n "${do_install:-}" ]; then
  cmake --install ${build_build_dir} --verbose
fi

#### test archive

# The tester images extract this at / and drive ci/internal/test.sh out of the
# source tree it carries, so the archive keeps absolute paths. The build tree
# and the original source tree are archived as two separate entries.
# _venv is excluded; ci/internal/test.sh builds its own.
tar --exclude=.git --exclude=_venv -czf ${BUILD_ARTIFACTS_DIR}/${tests_tarball_name} ${build_src_dir} ${src_dir}
du -sh ${BUILD_ARTIFACTS_DIR}/${tests_tarball_name}
