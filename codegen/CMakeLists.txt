# Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

## Codegen adds the following build targets
#    libcudacxx.atomics.codegen          (NV : produce cuda_ptx_generated.h)
#    libcudacxx.atomics.codegen.install  (NV : copy into include/.../functions/)
#    libcudacxx.atomics.codegen.hip      (HIP: produce atomic_hip_generated.h)
#    libcudacxx.atomics.codegen.hip.install (HIP: copy into include/.../functions/)
## Test targets:
#    libcudacxx.test.atomics.codegen.diff       (NV  drift guard)
#    libcudacxx.test.atomics.codegen.hip.diff   (HIP drift guard)

# NOTE(HIP/AMD): the parent CMakeLists.txt only pulls
# 'cmake/CCCLGetDependencies.cmake' (which defines cccl_get_fmt()) inside the
# LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON branch, AFTER add_subdirectory(codegen)
# has already been processed. In a standalone libhipcxx build with
# '-Dlibcudacxx_ENABLE_CODEGEN=ON' that ordering left the macro undefined here.
# Pull the helpers in directly if necessary so the codegen subtree configures
# regardless of the test-enable flag.
if (NOT COMMAND cccl_get_fmt)
  include("${libcudacxx_SOURCE_DIR}/cmake/CCCLGetDependencies.cmake")
endif()

add_executable(codegen EXCLUDE_FROM_ALL codegen.cpp)

target_compile_features(codegen PRIVATE cxx_std_20)

set(
  atomic_generated_output
  "${libcudacxx_BINARY_DIR}/codegen/cuda_ptx_generated.h"
)
set(
  atomic_install_location
  "${libcudacxx_SOURCE_DIR}/include/cuda/std/__atomic/functions"
)

add_custom_target(
  libcudacxx.atomics.codegen
  COMMAND codegen "${atomic_generated_output}"
  BYPRODUCTS "${atomic_generated_output}"
)

add_custom_target(
  libcudacxx.atomics.codegen.install
  # gersemi: off
  COMMAND
    "${CMAKE_COMMAND}" -E copy
      "${atomic_generated_output}"
      "${atomic_install_location}/cuda_ptx_generated.h"
  # gersemi: on
  DEPENDS libcudacxx.atomics.codegen
  BYPRODUCTS "${atomic_install_location}/cuda_ptx_generated.h"
)

add_test(
  NAME libcudacxx.test.atomics.codegen.diff
  # gersemi: off
  COMMAND
    "${CMAKE_COMMAND}" -E compare_files
      "${atomic_install_location}/cuda_ptx_generated.h"
      "${atomic_generated_output}"
  # gersemi: on
)

set_tests_properties(
  libcudacxx.test.atomics.codegen.diff
  PROPERTIES REQUIRED_FILES "${atomic_generated_output}"
)

# NOTE(HIP/AMD): mirror the NV-side regenerate / install / diff-test triple
# for the HIP path. The HIP emitter shares the same 'codegen' executable
# (selected via '--hip') and writes to a sibling output path, so the only
# additional CMake state is the output / install location and one extra
# custom target + ctest entry. The diff-test is the drift guard --
# 'libcudacxx.test.atomics.codegen.hip.diff' fails as soon as a developer
# edits 'atomic_hip_generated.h' by hand without first updating the
# generator template in 'codegen/generators/hip_intrinsics.h'.
set(
  atomic_hip_generated_output
  "${libcudacxx_BINARY_DIR}/codegen/atomic_hip_generated.h"
)

# License text emitted into the top of 'atomic_hip_generated.h'. Lives as a
# sibling text file (rather than inside 'hip_intrinsics.h' as a raw-string
# literal) so the .h template carries at most one license header for its own
# source code. 'FormatHipHeader' reads this file at codegen time via the
# LICENSE_FILE env var. Updating the emitted license is a one-file edit
# followed by a regen ('cmake --build --target libcudacxx.atomics.codegen.hip.install').
set(
  atomic_hip_generated_license
  "${CMAKE_CURRENT_SOURCE_DIR}/generators/atomic_hip_generated.license.txt"
)

add_custom_target(
  libcudacxx.atomics.codegen.hip
  # gersemi: off
  # NOTE: '$<TARGET_FILE:codegen>' (instead of bare 'codegen') is required
  # because we wrap the invocation in 'cmake -E env LICENSE_FILE=...', which
  # bypasses CMake's automatic target-name -> binary-path resolution that
  # add_custom_target's COMMAND would otherwise apply. Without TARGET_FILE,
  # 'env' would look 'codegen' up on PATH and fail with 'no such file or
  # directory' on every machine where the codegen build dir isn't already
  # in PATH (i.e. every machine).
  COMMAND
    ${CMAKE_COMMAND} -E env
      LICENSE_FILE=${atomic_hip_generated_license}
    $<TARGET_FILE:codegen> --hip "${atomic_hip_generated_output}"
  # gersemi: on
  BYPRODUCTS "${atomic_hip_generated_output}"
)

add_custom_target(
  libcudacxx.atomics.codegen.hip.install
  # gersemi: off
  COMMAND
    "${CMAKE_COMMAND}" -E copy
      "${atomic_hip_generated_output}"
      "${atomic_install_location}/atomic_hip_generated.h"
  # gersemi: on
  DEPENDS libcudacxx.atomics.codegen.hip
  BYPRODUCTS "${atomic_install_location}/atomic_hip_generated.h"
)

add_test(
  NAME libcudacxx.test.atomics.codegen.hip.diff
  # gersemi: off
  COMMAND
    "${CMAKE_COMMAND}" -E compare_files
      "${atomic_install_location}/atomic_hip_generated.h"
      "${atomic_hip_generated_output}"
  # gersemi: on
)

set_tests_properties(
  libcudacxx.test.atomics.codegen.hip.diff
  PROPERTIES REQUIRED_FILES "${atomic_hip_generated_output}"
)
