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

# Expose libcudacxx_SOURCE_DIR for parity with the upstream libcudacxx
# project name; lets all libhipcxx CMake code use the upstream variable
# name so vendored cmake/Libcudacxx*.cmake helpers stay byte-identical.
set(libcudacxx_SOURCE_DIR "${libhipcxx_SOURCE_DIR}")

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
option(
  libcudacxx_ENABLE_CODEGEN
  "Enable libcudacxx's atomics backend codegen and tests."
  OFF
)
if (libcudacxx_ENABLE_CODEGEN)
  add_subdirectory(codegen)
endif()

# Note that this currently returns and skips the rest of the build
# system.
option(libhipcxx_ENABLE_CMAKE_TESTS "Enable ctest-based testing." ON)
if (libhipcxx_ENABLE_CMAKE_TESTS)
  add_subdirectory(cmake/test)
endif()

set(CMAKE_MODULE_PATH "${libcudacxx_SOURCE_DIR}/cmake")
set(LLVM_PATH "${libcudacxx_SOURCE_DIR}" CACHE STRING "" FORCE)

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

if(WIN32)
  message(STATUS "Checking ${CMAKE_CURRENT_SOURCE_DIR}/include/libhipcxx/hip for symlink")
  # Delete hip symlink if not already a copy
  libhipcxx_path_is_link_like("${CMAKE_CURRENT_SOURCE_DIR}/include/libhipcxx/hip" _hip_path_is_link)
  if(_hip_path_is_link)
    file(REMOVE_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/include/libhipcxx/hip")
    message(STATUS "Symlink removed: ${CMAKE_CURRENT_SOURCE_DIR}/include/libhipcxx/hip")
  endif()
  # Delete test symlink if not already a copy
  libhipcxx_path_is_link_like("${CMAKE_CURRENT_SOURCE_DIR}/test" _test_path_is_link)
  if(_test_path_is_link)
    file(REMOVE_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/test")
    message(STATUS "Symlink removed: ${CMAKE_CURRENT_SOURCE_DIR}/test")
  endif()
  # Delete test symlink if not already a copy
  libhipcxx_path_is_link_like("${CMAKE_CURRENT_SOURCE_DIR}/utils" _utils_path_is_link)
  if(_utils_path_is_link)
    file(REMOVE_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/utils")
    message(STATUS "Symlink removed: ${CMAKE_CURRENT_SOURCE_DIR}/utils")
  endif()
  # Create a copy of cuda as hip folder
  message(STATUS "Copying cuda folder to hip to allow for hip/cuda aliasing.")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/libhipcxx/hip")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/include/libhipcxx/hip/")
  # Create a copy of .upstream_test/test as test folder
  message(STATUS "Copying .upstream_tests/test folder to test.")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/.upstream-tests/test/" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/test/")
  # Create a copy of .upstream_test/utils as utils folder
  message(STATUS "Copying .upstream_tests/utils folder to utils.")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/utils")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/.upstream-tests/utils/" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/utils/")
endif()

# As part of the TheRock testing process, we need the source directory within the build folder
# since we need to configure, build and run the tests from within the tester using lit.
# The reason for this is that lit cannot seperately generate binaries in the build step
# and execute them in the tester but directly executes and deletes any test binaries during the build.
if(DEFINED THEROCK_TOOLCHAIN_ROOT)
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()

option(LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS "Enable libcu++ tests." ON)
if (LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS)
  enable_testing()

  # Pull in the CCCL infrastructure that our Libcudacxx*.cmake helpers
  # depend on. In the upstream CCCL workflow these are loaded by the parent
  # CCCL CMakeLists.txt before libcudacxx is added as a subdirectory; since
  # libhipcxx is a standalone project we load them here directly.
  # - AppendOptionIfAvailable.cmake provides append_option_if_available()
  # - CCCLBuildCompilerTargets.cmake provides cccl_build_compiler_interface()
  #   and cccl_build_compiler_targets() which creates the
  #   cccl.compiler_interface_cppXX targets that libcudacxx links against.
  # - CCCLConfigureTarget.cmake provides cccl_configure_target() used by the
  #   header-testing helpers below.
  include(cmake/AppendOptionIfAvailable.cmake)
  include(cmake/CCCLBuildCompilerTargets.cmake)
  include(cmake/CCCLConfigureTarget.cmake)
  include(cmake/CCCLGetDependencies.cmake)
  cccl_build_compiler_targets()

  # libcudacxx_build_compiler_targets() links libcudacxx.compiler_interface
  # against cccl.compiler_interface_cpp${CMAKE_CUDA_STANDARD}. When CUDA is
  # not enabled (HIP-only build) CMAKE_CUDA_STANDARD is empty and the link
  # resolves to a non-existent target. Default to 17 to mirror the upstream
  # libcudacxx CMakePresets baseline.
  if (NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
  endif()

  # Create the compiler targets with the common compile flags
  include(cmake/LibcudacxxBuildCompilerTargets.cmake)
  libcudacxx_build_compiler_targets()

  # Test all public and internal headers.
  # The Internal/PublicHeaderTesting helpers create per-header .cu library
  # targets that require enable_language(CUDA); only include them when CUDA
  # is enabled. The PublicHeaderTestingHost helper creates .cpp library
  # targets and works under both CUDA and HIP.
  if (LIBCUDACXX_ENABLE_CUDA)
    include(cmake/LibcudacxxInternalHeaderTesting.cmake)
    include(cmake/LibcudacxxPublicHeaderTesting.cmake)
  endif()
  include(cmake/LibcudacxxPublicHeaderTestingHost.cmake)

  add_subdirectory(test)
endif()

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
