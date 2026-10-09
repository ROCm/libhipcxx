# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import shutil

# Required settings
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm",
    "version_list_link": "https://rocm.docs.amd.com/projects/libhipcxx/en/latest/release.html",
    "repository_url": "https://github.com/ROCm/libhipcxx",
    "path_to_docs": "projects/libhipcxx/docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}


# This section turns on/off article info
setting_all_article_info = True
all_article_info_os = ["linux", "windows"]
all_article_info_author = ""

# Dynamically extract component version
#with open('../CMakeLists.txt', encoding='utf-8') as f:
#    pattern = (
#        r'.*\brocm_setup_version\(VERSION\s+([0-9.]+)[^0-9.]+'  # Update according to each component's CMakeLists.txt
#    )
#    match = re.search(pattern, f.read())
#    if not match:
#        raise ValueError("VERSION not found!")
version_number = "3.0.2"

# for PDF output on Read the Docs
project = "libhipcxx"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

exclude_patterns = [
    '_includes/**',
    # Not supported in libhipcxx, see reference/libhipcxx-limitations.rst.
    'libcudacxx/extended_api/synchronization_primitives/latch.rst',
    'libcudacxx/extended_api/synchronization_primitives/barrier.rst',
    'libcudacxx/extended_api/synchronization_primitives/barrier/**',
    'libcudacxx/extended_api/synchronization_primitives/counting_semaphore.rst',
    'libcudacxx/extended_api/synchronization_primitives/binary_semaphore.rst',
    'libcudacxx/extended_api/synchronization_primitives/pipeline.rst',
    'libcudacxx/extended_api/synchronization_primitives/pipeline/**',
    'libcudacxx/extended_api/asynchronous_operations.rst',
    'libcudacxx/extended_api/asynchronous_operations/**',
    'libcudacxx/extended_api/memory/aligned_size.rst',
    'libcudacxx/extended_api/memory/discard_memory.rst',
    'libcudacxx/extended_api/tma.rst',
    'libcudacxx/extended_api/tma/**',
    'libcudacxx/runtime/buffer.rst',
    'libcudacxx/tile.rst',
    'libcudacxx/extended_api/memory_access_properties.rst',
    'libcudacxx/extended_api/memory_access_properties/**',
    'libcudacxx/extended_api/warp.rst',
    'libcudacxx/extended_api/warp/**',
    'libcudacxx/extended_api/work_stealing.rst',
    # Contains CUDA-specific forward progress guarantees and execution model details.
    'libcudacxx/extended_api/execution_model.rst',
    # NVIDIA/libcudacxx-specific: the upstream release table maps libcu++ versions to CUDA
    # toolkit releases, and the changelog describes upstream releases. The libhipcxx version
    # and ABI facts live in reference/libhipcxx-conformance.rst.
    'libcudacxx/releases.rst',
    'libcudacxx/releases/**',
    # Upstream CCCL monorepo leftovers, unreachable from sphinx/_toc.yml.in. cpp.rst and
    # python.rst are landing pages for CUB, Thrust and the cuda.* Python packages, which
    # libhipcxx does not ship, and the code of conduct names the NVIDIA C++ Core Compute
    # Libraries community and its contact address. VERSION.md is a stale version string.
    'VERSION.md',
    'cpp.rst',
    'python.rst',
    'libcudacxx/contributing.rst',
    'libcudacxx/contributing/**',
]

# Generated at build time from sphinx/_toc.yml.in (rocm-docs-core). Keep it under
# _build/html so sphinx-autobuild does not watch and rebuild in a loop.
_build_toc_dir = os.path.join(os.path.dirname(__file__), "_build", "html")
os.makedirs(_build_toc_dir, exist_ok=True)
external_toc_template_path = "./sphinx/_toc.yml.in"
external_toc_path = "./_build/html/_toc.yml"

# rocm_docs regenerates _toc.yml in config-inited (priority 500) after
# sphinx-external-toc parses it (priority 900). Sync early so sidebar order
# matches _toc.yml.in on the first build after edits.
_toc_in = os.path.join(os.path.dirname(__file__), "sphinx", "_toc.yml.in")
_toc_out = os.path.join(_build_toc_dir, "_toc.yml")
if os.path.isfile(_toc_in):
    shutil.copy2(_toc_in, _toc_out)

# Optional: skip fetching projects.yaml from GitHub (see Makefile html-doc target).
# "Mappings" = intersphinx project URL map in rocm-docs-core's bundled data/projects.yaml.
if os.environ.get("ROCM_DOCS_USE_BUNDLED_MAPPINGS"):
    external_projects_remote_repository = ""

# Add more addtional package accordingly
extensions = [
    "rocm_docs",
]

html_title = f"{project} {version_number} documentation"

external_projects_current_project = "libhipcxx"

# Override external projects to avoid broken intersphinx mappings
external_projects = {}

# Disable problematic intersphinx mappings
intersphinx_mapping = {}

# Make intersphinx failures non-fatal
suppress_warnings = ['app.add_node', 'app.add_directive']

# Configure Sphinx to continue on intersphinx errors
nitpicky = False
