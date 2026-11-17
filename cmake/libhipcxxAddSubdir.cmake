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

# This effectively does a `find_package` actually going through the find_package
# machinery. Using `find_package` works for the first configure, but creates
# inconsistencies during subsequent configurations when using CPM..
#
# More details are in the discussion at
# https://github.com/NVIDIA/libcudacxx/pull/242#discussion_r794003857
#
# Called outside find_package(), from add_subdirectory and from the top-level
# build. The version file writes PACKAGE_VERSION and the other find_package
# result variables; a function scope drops those on return so they do not
# replace the caller's packaging version. The INTERFACE target and the
# include-dir cache entries stay visible.
function(_libhipcxx_define_from_source)
  include(${libcudacxx_SOURCE_DIR}/lib/cmake/libhipcxx/libhipcxx-config-version.cmake)
  include(${libcudacxx_SOURCE_DIR}/lib/cmake/libhipcxx/libhipcxx-config.cmake)
endfunction()
_libhipcxx_define_from_source()
