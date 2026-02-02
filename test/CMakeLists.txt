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

find_package(Python COMPONENTS Interpreter)
if (NOT Python_Interpreter_FOUND)
  message(
    FATAL_ERROR
    "Failed to find python interpreter, which is required for running tests and building a libcu++ static library."
  )
endif()

# Determine the host triple to avoid invoking `${CXX} -dumpmachine`.
include(GetHostTriple)
get_host_triple(LLVM_INFERRED_HOST_TRIPLE)

set(
  LLVM_HOST_TRIPLE
  "${LLVM_INFERRED_HOST_TRIPLE}"
  CACHE STRING
  "Host on which LLVM binaries will run"
)

# By default, we target the host, but this can be overridden at CMake
# invocation time.
set(
  LLVM_DEFAULT_TARGET_TRIPLE
  "${LLVM_HOST_TRIPLE}"
  CACHE STRING
  "Default target for which LLVM will generate code."
)
set(TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}")
message(STATUS "LLVM host triple: ${LLVM_HOST_TRIPLE}")
message(STATUS "LLVM default target triple: ${LLVM_DEFAULT_TARGET_TRIPLE}")

set(LIT_EXTRA_ARGS "" CACHE STRING "Use for additional options (e.g. -j12)")
find_program(LLVM_DEFAULT_EXTERNAL_LIT lit)
set(LLVM_LIT_ARGS "-sv ${LIT_EXTRA_ARGS}")

# Libcudacxx's main lit tests
add_subdirectory(libcudacxx)

add_subdirectory(cmake)

# Set appropriate warning levels for MSVC/sane
if ("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
  # CUDA 11.5 and down do not support '-use-local-env'
  if (MSVC)
    set(
      headertest_warning_levels_device
      -Xcompiler=/W4
      -Xcompiler=/WX
      -Wno-deprecated-gpu-targets
    )
    if ("${CMAKE_CUDA_COMPILER_VERSION}" GREATER_EQUAL "11.6.0")
      list(APPEND headertest_warning_levels_device --use-local-env)
    endif()
  else()
    set(
      headertest_warning_levels_device
      -Wall
      -Werror
      all-warnings
      -Wno-deprecated-gpu-targets
    )
  endif()
  # Set warnings for Clang as device compiler
elseif ("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
  set(
    headertest_warning_levels_device
    -Wall
    -Werror
    -Wno-unknown-cuda-version
    -Xclang=-fcuda-allow-variadic-functions
  )
  # If the CMAKE_CUDA_COMPILER is unknown, try to use gcc style warnings
else()
  set(headertest_warning_levels_device -Wall -Werror)
endif()

# Set raw host/device warnings
if (MSVC)
  set(headertest_warning_levels_host /W4 /WX)
else()
  set(headertest_warning_levels_host -Wall -Werror)
endif()

# Build HIPRTC/NVRTC compiler wrapper if needed
if (LIBHIPCXX_TEST_WITH_HIPRTC)
  add_subdirectory(utils/amd/hiprtc)
elseif (LIBCUDACXX_TEST_WITH_NVRTC)
  add_subdirectory(utils/nvidia/nvrtc)
endif()

add_subdirectory(nvtarget)
add_subdirectory(atomic_codegen)
