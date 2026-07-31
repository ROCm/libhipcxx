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

option(libhipcxx_ENABLE_INSTALL_RULES
  "Enable installation of libhipcxx" ${LIBCUDACXX_TOPLEVEL_PROJECT}
)

if (NOT libhipcxx_ENABLE_INSTALL_RULES)
  return()
endif()

# Bring in CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# NOTE(HIP): We explicitly specify FILES_MATCHING with PATTERN *, as otherwise
# ROCm-cmake may generate an invalid install(*) command where COMPONENT comes
# after PATTERN, which is not valid in CMake

# NOTE(HIP): All libhipcxx headers are installed *under* '<inc>/libhipcxx/'
# instead of directly into '<inc>/'. This keeps the whole payload self-contained
# in a single subtree (so a single '-I<inc>/libhipcxx' resolves <cuda/...>,
# <nv/...>, <amd/...> and <hip/...>) and avoids polluting/colliding with the
# top-level '<inc>/cuda' of a CUDA Toolkit or ROCm's own headers. Consumers must
# pull in libhipcxx via find_package(libhipcxx) + libhipcxx::libhipcxx (which
# carries the correct include dir); bare '#include <cuda/...>' off the default
# '<inc>' search path no longer resolves.
#
# The subtree name is exposed as a cache variable so downstream packagers can
# rename it (or set it empty to install straight into '<inc>/') from the command
# line without editing these rules, e.g. -DLIBHIPCXX_INSTALL_INCLUDE_SUBDIR=foo.
# LIBHIPCXX_INSTALL_INCLUDEDIR is the single source of truth for the header
# destinations below *and* for the consumer-side probe path baked into
# libhipcxx-header-search.cmake.in (via configure_file), so the install layout
# and the find_package() detection can never drift apart.
set(LIBHIPCXX_INSTALL_INCLUDE_SUBDIR "libhipcxx"
  CACHE STRING
  "Subdirectory under the install includedir that receives libhipcxx headers (empty = install directly into <includedir>)"
)

if(LIBHIPCXX_INSTALL_INCLUDE_SUBDIR)
  set(LIBHIPCXX_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/${LIBHIPCXX_INSTALL_INCLUDE_SUBDIR}")
else()
  set(LIBHIPCXX_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

# Libhipcxx headers
rocm_install(DIRECTORY "${libhipcxx_SOURCE_DIR}/include/cuda"
  DESTINATION "${LIBHIPCXX_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
  PATTERN *
  PATTERN CMakeLists.txt EXCLUDE
)
rocm_install(DIRECTORY "${libhipcxx_SOURCE_DIR}/include/nv"
  DESTINATION "${LIBHIPCXX_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
  PATTERN *
  PATTERN CMakeLists.txt EXCLUDE
)
rocm_install(DIRECTORY "${libhipcxx_SOURCE_DIR}/include/amd"
  DESTINATION "${LIBHIPCXX_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
  PATTERN *
  PATTERN CMakeLists.txt EXCLUDE
)

# Copy libhipcxx headers into the hip folder additionally to provide the
# hip/cuda aliasing (i.e. '#include <hip/std/...>'). This destination already
# lives under the libhipcxx include subtree, so it tracks the subtree move above.
# Note: we can't use symlinks here, as this would
# break builds of packages like hipDF which
# create a Python wheel with setuptools.
rocm_install(DIRECTORY "${libhipcxx_SOURCE_DIR}/include/cuda/"
  DESTINATION "${LIBHIPCXX_INSTALL_INCLUDEDIR}/hip"
  FILES_MATCHING
  PATTERN *
  PATTERN CMakeLists.txt EXCLUDE
)

# Libcudacxx cmake package
rocm_install(DIRECTORY "${libhipcxx_SOURCE_DIR}/lib/cmake/libhipcxx"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
  FILES_MATCHING
  PATTERN *
  REGEX .*header-search.cmake.* EXCLUDE
)

set(install_location "${CMAKE_INSTALL_LIBDIR}/cmake/libhipcxx")
# Transform to a list of directories, replace each directory with "../"
# and convert back to a string
string(REGEX REPLACE "/" ";" from_install_prefix "${install_location}")
list(TRANSFORM from_install_prefix REPLACE ".+" "../")
list(JOIN from_install_prefix "" from_install_prefix)

# Need to configure a file to store CMAKE_INSTALL_INCLUDEDIR
# since it can be defined by the user. This is common to work around collisions
# with the CTK installed headers.
configure_file("${libhipcxx_SOURCE_DIR}/lib/cmake/libhipcxx/libhipcxx-header-search.cmake.in"
  "${libhipcxx_BINARY_DIR}/lib/cmake/libhipcxx/libhipcxx-header-search.cmake"
  @ONLY
)
rocm_install(FILES "${libhipcxx_BINARY_DIR}/lib/cmake/libhipcxx/libhipcxx-header-search.cmake"
  DESTINATION "${install_location}"
)
