# MIT License
#
# Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

add_custom_target(libcudacxx.test.atomics.ptx)

find_program(filecheck "FileCheck" HINTS /opt/rocm/lib/llvm/bin)

if (filecheck)
  message("-- ${filecheck} found... building atomic codegen tests")
else()
  return()
endif()

# NOTE(HIP/AMD): the codegen tests run on both NVIDIA (cuobjdump
# --dump-ptx + FileCheck against PTX) and HIP (clang -emit-llvm +
# FileCheck against LLVM-IR atomicrmw / cmpxchg / load atomic /
# store atomic). The .cu sources are SHARED between the two paths --
# the upstream-style '/* SM8X-LABEL: ... */' comment block stays as
# the NV check; we add an additional 'HIP_IR-LABEL: ...' block at
# the end of the same comment fenced by an explicit
# '; ----- AMD/HIP additions below -----' divider. The driver
# 'dump_and_check.bash' takes a 4th argument NV|HIP that selects
# the disassembly toolchain. The umbrella target name
# 'libcudacxx.test.atomics.ptx' is preserved as-is on both paths so
# that 'cmake --build --target libcudacxx.test.atomics.ptx' keeps
# working on every backend (and so upstream merges that touch the
# target name keep auto-merging).
find_program(bash "bash" REQUIRED)
if (LIBCUDACXX_ENABLE_CUDA)
  find_program(cuobjdump "cuobjdump" REQUIRED)
  set(_codegen_platform "NV")
  set(_codegen_check_prefix "SM8X")
elseif (LIBCUDACXX_ENABLE_HIP)
  set(_codegen_platform "HIP")
  set(_codegen_check_prefix "HIP_IR")
  # Pick the first entry of CMAKE_HIP_ARCHITECTURES (which the
  # surrounding libhipcxx build already sets up via AMDGPU_TARGETS /
  # GPU_TARGETS / explicit -DCMAKE_HIP_ARCHITECTURES). Fall back to
  # gfx90a if nothing is set. Override at configure time with
  # -DLIBCUDACXX_ATOMIC_CODEGEN_HIP_ARCH=gfxNN.
  if (NOT DEFINED LIBCUDACXX_ATOMIC_CODEGEN_HIP_ARCH)
    if (CMAKE_HIP_ARCHITECTURES)
      list(GET CMAKE_HIP_ARCHITECTURES 0 LIBCUDACXX_ATOMIC_CODEGEN_HIP_ARCH)
    else()
      set(LIBCUDACXX_ATOMIC_CODEGEN_HIP_ARCH "gfx90a")
    endif()
  endif()
  message(STATUS "atomic codegen tests (HIP): targeting ${LIBCUDACXX_ATOMIC_CODEGEN_HIP_ARCH}")
else()
  return()
endif()

set(atomic_codegen_cuda_arch 80)

set(libcudacxx_atomic_codegen_tests)
if (NOT "NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  file(GLOB libcudacxx_atomic_codegen_tests "*.cu")
endif()

# For every atomic API compile the TU and check if the SASS/PTX (NV) or
# LLVM-IR (HIP) matches the expected pattern in the .cu file's '/* ... */'
# trailing comment.
foreach (test_path IN LISTS libcudacxx_atomic_codegen_tests)
  cmake_path(GET test_path FILENAME test_file)
  cmake_path(REMOVE_EXTENSION test_file LAST_ONLY OUTPUT_VARIABLE test_name)

  if (LIBCUDACXX_ENABLE_CUDA)
    # === NV path: existing upstream behaviour, unchanged ===
    add_library(atomic_codegen_${test_name} STATIC "${test_path}")

    set_target_properties(
      atomic_codegen_${test_name}
      PROPERTIES
        CUDA_ARCHITECTURES "${atomic_codegen_cuda_arch}"
        COMPILE_DEFINITIONS "_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE=1"
    )

    # Clang stopped emitting PTX in clang20. Add flags to re-enable it.
    if (
      CMAKE_CUDA_COMPILER_ID STREQUAL Clang
      AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 20
    )
      target_compile_options(
        atomic_codegen_${test_name}
        PRIVATE "--cuda-include-ptx=sm_${atomic_codegen_cuda_arch}"
      )
    endif()

    target_compile_options(atomic_codegen_${test_name} PRIVATE "-Wno-comment")

    ## Important for testing the local headers
    target_include_directories(
      atomic_codegen_${test_name}
      PRIVATE "${libcudacxx_SOURCE_DIR}/include"
    )
    add_dependencies(libcudacxx.test.atomics.ptx atomic_codegen_${test_name})

    # Add output path to object directory
    add_custom_command(
      TARGET libcudacxx.test.atomics.ptx
      POST_BUILD
      # gersemi: off
      COMMAND
        ${CMAKE_COMMAND} -E env
          FILECHECK=${filecheck}
        "${CMAKE_CURRENT_SOURCE_DIR}/dump_and_check.bash"
          $<TARGET_FILE:atomic_codegen_${test_name}>
          "${test_path}"
          ${_codegen_check_prefix}
          ${_codegen_platform}
      # gersemi: on
    )

  elseif (LIBCUDACXX_ENABLE_HIP)
    # === HIP path: 'clang -x hip -emit-llvm' directly inside the script,
    # no static-lib intermediate, no cuobjdump. The script compiles the
    # .cu to a tempfile .ll using a baked-in HIP flag set (matching the
    # libhipcxx test infra: -include force_include_hip.h, -I include,
    # -I test/support, etc.), then FileChecks the IR against the .cu's
    # '; HIP_IR:' lines.
    add_custom_target(
      atomic_codegen_${test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
          LIBCUDACXX_SOURCE_DIR=${libcudacxx_SOURCE_DIR}
          HIP_IR_ARCH=${LIBCUDACXX_ATOMIC_CODEGEN_HIP_ARCH}
          HIP_IR_CXX=${CMAKE_HIP_COMPILER}
          FILECHECK=${filecheck}
        "${CMAKE_CURRENT_SOURCE_DIR}/dump_and_check.bash"
          "${test_path}"
          "${test_path}"
          ${_codegen_check_prefix}
          ${_codegen_platform}
      VERBATIM
    )
    add_dependencies(libcudacxx.test.atomics.ptx atomic_codegen_${test_name})
  endif()
endforeach()
