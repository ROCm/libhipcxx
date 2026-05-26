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

# libcudacxx_filter_backend_headers(out_list
#                                   BACKEND <name>
#                                   SKIP_DIRS    <dir1> [<dir2> ...]
#                                   ALLOWLIST_HEADERS <hdr1> [<hdr2> ...])
#
# Filter the in-place 'out_list' (name of a variable in the caller's scope)
# down to the headers that should be exercised by the backend-specific
# header-testing sweep:
#
#   * Drop every header whose path starts with one of the SKIP_DIRS
#     directories. Mass-skips the upstream feature surfaces that have no
#     HIP-portable implementation (cuda/__barrier, cuda/__semaphore, etc).
#   * Re-add any header listed in ALLOWLIST_HEADERS. Used for individual
#     headers under a skipped directory that DO have a HIP-portable
#     implementation today (e.g. specific cuda::ptx::* wrappers under
#     cuda/__ptx/ that ship a software emulation -- see the consolidated
#     NOTE in <cuda/__ptx/ptx_helper_functions.h>).
#
# BACKEND is currently informational (used in the status message); kept
# as a structured argument so a future second backend can reuse the
# helper without changing the public shape.
#
# Rationale: this hard-coded block-list/allow-list approach replaces an
# earlier per-header regex-match inside the test-creation function. The
# regex pattern was brittle: a future header under one of the skipped
# directories that DOES have HIP support would have been silently filtered
# out with no signal at configure time. The block-list approach surfaces
# new files as build breakages (a fresh upstream header under
# 'cuda/__barrier/' won't be skipped automatically -- you'll see the
# compile failure and explicitly decide whether to add it to SKIP_DIRS,
# ALLOWLIST_HEADERS, or wire up the missing HIP support). This costs
# slightly more during upgrades, but the loss-of-coverage risk is the
# reverse of what we want.
#
# Adopted from gpinkert_amdeng's suggestion on PR #217 (g17).
function(libcudacxx_filter_backend_headers out_list)
  set(opts "")
  set(one  "BACKEND")
  set(many "SKIP_DIRS;ALLOWLIST_HEADERS")
  cmake_parse_arguments(arg "${opts}" "${one}" "${many}" ${ARGN})

  # Trailing '(/|$)' makes the SKIP_DIRS list accept both directory prefixes
  # (matches e.g. 'cuda/__barrier/foo.h') AND exact umbrella-header file
  # names without an extension or subdirectory (matches e.g. the public
  # umbrella header at 'cuda/std/barrier'). Lets one helper handle the
  # internal-headers sweep (subdirs) and the public-headers sweep
  # (umbrella files) with the same SKIP_DIRS shape on the caller side.
  list(JOIN arg_SKIP_DIRS "|" _re)
  set(_re "^(${_re})(/|$)")

  set(_kept "")
  foreach (h IN LISTS ${out_list})
    if (NOT h MATCHES "${_re}" OR h IN_LIST arg_ALLOWLIST_HEADERS)
      list(APPEND _kept "${h}")
    endif()
  endforeach()
  set(${out_list} "${_kept}" PARENT_SCOPE)
endfunction()
