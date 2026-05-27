# Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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

# 3.15 is the minimum.
# 3.17 for NVC++.
# 3.18 for C++17 + CUDA and clang-cuda support.
# 3.21 is for HIP
cmake_minimum_required(VERSION 3.21)

# Determine whether libhipcxx is the top-level project or included into
# another project via add_subdirectory().
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(LIBCUDACXX_TOPLEVEL_PROJECT ON)
endif()

if (LIBCUDACXX_TOPLEVEL_PROJECT)
  cmake_minimum_required(VERSION 3.21)
endif()

set(PACKAGE_NAME libhipcxx)
set(PACKAGE_VERSION 11.0)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")

project(libhipcxx LANGUAGES CXX VERSION 3.0.2)

# rocm-cmake contains common cmake code for rocm projects to help
# setup and install
include(cmake/FindROCMCmake.cmake)
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
# include(ROCMPackageConfigHelpers)
# include(ROCMInstallSymlinks)
# include(ROCMHeaderWrapper)
# include(ROCMCheckTargetIds)
# include(ROCMClients)
option(libhipcxx_ENABLE_INSTALL_RULES "Enable installation of libcu++" ${LIBCUDACXX_TOPLEVEL_PROJECT})
if (libhipcxx_ENABLE_INSTALL_RULES)
  include(cmake/libhipcxxInstallRules.cmake)
endif()

if (NOT LIBCUDACXX_TOPLEVEL_PROJECT)
  include(cmake/libhipcxxAddSubdir.cmake)
  return()
endif()

# Must be called in root CMakeLists.txt
include(CTest)
enable_testing()

# Add codegen module
option(libhipcxx_ENABLE_CODEGEN "Enable libhipcxx's atomics backend codegen and tests." OFF)
if (libhipcxx_ENABLE_CODEGEN)
  add_subdirectory(codegen)
endif()

# Note that this currently returns and skips the rest of the build
# system.
option(libhipcxx_ENABLE_CMAKE_TESTS "Enable ctest-based testing." ON)
if (libhipcxx_ENABLE_CMAKE_TESTS)
  add_subdirectory(cmake/test)
endif()

set(CMAKE_MODULE_PATH "${libhipcxx_SOURCE_DIR}/cmake")
set(LLVM_PATH "${libhipcxx_SOURCE_DIR}" CACHE STRING "" FORCE)

# Configuration options.
option(LIBCUDACXX_ENABLE_CUDA "Enable the CUDA language support." OFF)
option(LIBCUDACXX_ENABLE_HIP "Enable the HIP language support." ON)

if (LIBCUDACXX_ENABLE_CUDA AND LIBCUDACXX_ENABLE_HIP)
  message(FATAL_ERROR "HIP and CUDA compilers cannot be used at the same time, CMake will exit." )
endif()

if (LIBCUDACXX_ENABLE_CUDA)
  message(FATAL_ERROR "CUDA backend support is currently not supported. CMake will exit.")
endif()

if (LIBCUDACXX_ENABLE_CUDA)
  enable_language(CUDA)
elseif (LIBCUDACXX_ENABLE_HIP)
  # sync variables used by HIP CMake package etc.
  if ("${AMDGPU_TARGETS}" STREQUAL "")
    set(AMDGPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
  elseif(NOT "${AMDGPU_TARGETS}" STREQUAL "${CMAKE_HIP_ARCHITECTURES}")
    message(WARNING "mismatch between CMAKE_HIP_ARCHITECTURES and AMDGPU_TARGETS! Using AMDGPU_TARGETS! CMAKE_HIP_ARCHITECTURES='${CMAKE_HIP_ARCHITECTURES}', AMDGPU_TARGETS='${AMDGPU_TARGETS}'")
    set(CMAKE_HIP_ARCHITECTURES ${AMDGPU_TARGETS})
  endif()
  if ("${GPU_TARGETS}" STREQUAL "")
    set(GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
  elseif(NOT "${GPU_TARGETS}" STREQUAL "${CMAKE_HIP_ARCHITECTURES}")
    message(WARNING "mismatch between CMAKE_HIP_ARCHITECTURES and GPU_TARGETS! Using GPU_TARGETS! CMAKE_HIP_ARCHITECTURES='${CMAKE_HIP_ARCHITECTURES}', GPU_TARGETS='${GPU_TARGETS}'")
    set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
  endif()

  find_package(hip REQUIRED)
  enable_language(HIP)
endif ()

# For Windows we should avoid symlinks by copying the file content (test, utils, and include/hip folder)
function(libhipcxx_path_is_link_like path out_var)
  if(IS_SYMLINK "${path}")
    set(${out_var} TRUE PARENT_SCOPE)
    return()
  endif()

  if(NOT WIN32 OR NOT EXISTS "${path}")
    set(${out_var} FALSE PARENT_SCOPE)
    return()
  endif()

  if(NOT IS_DIRECTORY "${path}")
    set(${out_var} TRUE PARENT_SCOPE)
    return()
  endif()

  get_filename_component(_abs_path "${path}" ABSOLUTE)
  file(REAL_PATH "${path}" _real_path)
  file(TO_CMAKE_PATH "${_abs_path}" _abs_path_norm)
  file(TO_CMAKE_PATH "${_real_path}" _real_path_norm)
  string(REGEX REPLACE "/$" "" _abs_path_norm "${_abs_path_norm}")
  string(REGEX REPLACE "/$" "" _real_path_norm "${_real_path_norm}")
  string(TOLOWER "${_abs_path_norm}" _abs_path_lc)
  string(TOLOWER "${_real_path_norm}" _real_path_lc)

  if(NOT _abs_path_lc STREQUAL _real_path_lc)
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

# OFF (default): no source-tree modification. On Win32 without
# LIBHIPCXX_COPY_TO_BUILD this requires working git symlinks in the
# source checkout (developer mode / core.symlinks=true). ON: legacy
# in-source fixup.
option(LIBHIPCXX_WIN32_REMOVE_SRC_SYMLINKS
  "On Win32, resolve symlinks in the source tree (modifies git-tracked files)." OFF)

# ON: copy the source tree into the build folder (TheRock tester needs
# this to re-configure/build/run lit from the build artifact). On Win32
# the copy is also the symlink-fixup target when REMOVE_SRC_SYMLINKS is OFF.
option(LIBHIPCXX_COPY_TO_BUILD
  "Copy source tree into build folder (set by TheRock; required for the tester pipeline)." OFF)

if(LIBHIPCXX_COPY_TO_BUILD)
  get_filename_component(_libhipcxx_src_name "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  set(_libhipcxx_build_src_copy "${CMAKE_CURRENT_BINARY_DIR}/${_libhipcxx_src_name}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# Win32: assemble fixup targets (source, build copy, both, or none).
set(_libhipcxx_win32_fixup_dirs)
if(WIN32)
  if(LIBHIPCXX_WIN32_REMOVE_SRC_SYMLINKS)
    list(APPEND _libhipcxx_win32_fixup_dirs "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  if(LIBHIPCXX_COPY_TO_BUILD)
    list(APPEND _libhipcxx_win32_fixup_dirs "${_libhipcxx_build_src_copy}")
  endif()
endif()

foreach(_dir IN LISTS _libhipcxx_win32_fixup_dirs)
  message(STATUS "Resolving Win32 symlinks in: ${_dir}")
  foreach(_p "include/libhipcxx/hip" "test" "utils")
    libhipcxx_path_is_link_like("${_dir}/${_p}" _is_link)
    if(_is_link)
      file(REMOVE_RECURSE "${_dir}/${_p}")
    endif()
  endforeach()
  file(MAKE_DIRECTORY "${_dir}/include/libhipcxx/hip")
  file(COPY "${_dir}/include/cuda/"           DESTINATION "${_dir}/include/libhipcxx/hip/")
  file(MAKE_DIRECTORY "${_dir}/test")
  file(COPY "${_dir}/.upstream-tests/test/"   DESTINATION "${_dir}/test/")
  file(MAKE_DIRECTORY "${_dir}/utils")
  file(COPY "${_dir}/.upstream-tests/utils/"  DESTINATION "${_dir}/utils/")
endforeach()

option(LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS "Enable libhip++ tests." ON)
if (LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS)
  enable_testing()

  include(FindPythonInterp)
  if (NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR
      "Failed to find python interpreter, which is required for running tests and "
      "building a libcu++ static library.")
  endif ()

  # Determine the host triple to avoid invoking `${CXX} -dumpmachine`.
  include(GetHostTriple)
  get_host_triple(LLVM_INFERRED_HOST_TRIPLE)

  set(LLVM_HOST_TRIPLE "${LLVM_INFERRED_HOST_TRIPLE}" CACHE STRING
      "Host on which LLVM binaries will run")

  # By default, we target the host, but this can be overridden at CMake
  # invocation time.
  set(LLVM_DEFAULT_TARGET_TRIPLE "${LLVM_HOST_TRIPLE}" CACHE STRING
    "Default target for which LLVM will generate code." )
  set(TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}")
  message(STATUS "LLVM host triple: ${LLVM_HOST_TRIPLE}")
  message(STATUS "LLVM default target triple: ${LLVM_DEFAULT_TARGET_TRIPLE}")

  set(LIT_EXTRA_ARGS "" CACHE STRING "Use for additional options (e.g. -j12)")
  find_program(LLVM_DEFAULT_EXTERNAL_LIT lit)
  set(LLVM_LIT_ARGS "-sv ${LIT_EXTRA_ARGS}")

  # On Win32 with the build copy available and no source modification, the
  # build copy is where the symlinks were resolved.
  if(WIN32 AND LIBHIPCXX_COPY_TO_BUILD AND NOT LIBHIPCXX_WIN32_REMOVE_SRC_SYMLINKS)
    set(libhipcxx_SOURCE_DIR "${_libhipcxx_build_src_copy}")
    add_subdirectory("${libhipcxx_SOURCE_DIR}/test" "${CMAKE_CURRENT_BINARY_DIR}/test")
  else()
    set(libhipcxx_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(test)
  endif()

  include(AddLLVM)
  get_property(LLVM_LIT_TESTSUITES          GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
  get_property(LLVM_LIT_PARAMS              GLOBAL PROPERTY LLVM_LIT_PARAMS)
  get_property(LLVM_LIT_DEPENDS             GLOBAL PROPERTY LLVM_LIT_DEPENDS)
  get_property(LLVM_LIT_EXTRA_ARGS          GLOBAL PROPERTY LLVM_LIT_EXTRA_ARGS)
  get_property(LLVM_ADDITIONAL_TEST_TARGETS GLOBAL PROPERTY LLVM_ADDITIONAL_TEST_TARGETS)
  get_property(LLVM_ADDITIONAL_TEST_DEPENDS GLOBAL PROPERTY LLVM_ADDITIONAL_TEST_DEPENDS)
  add_lit_target(check-all
    "Running all regression tests"
    ${LLVM_LIT_TESTSUITES}
    PARAMS ${LLVM_LIT_PARAMS}
    DEPENDS ${LLVM_LIT_DEPENDS} ${LLVM_ADDITIONAL_TEST_TARGETS}
    ARGS ${LLVM_LIT_EXTRA_ARGS}
  )
endif ()

# Package configuration
set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.TXT")
set(CPACK_RPM_PACKAGE_LICENSE "ASL 2.0 AND MIT")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

rocm_create_package(
  NAME libhipcxx
  DESCRIPTION "ROCm libhipcxx library"
  # FIXME(HIP/AMD): A maintainer string is required
  # Set to the following when available:
  # MAINTAINER "libhipcxx-maintainer@amd.com"
  MAINTAINER "AMD, Inc."
  HEADER_ONLY
)
