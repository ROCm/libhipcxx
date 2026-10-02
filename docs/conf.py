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
    "flavor": "generic",
    "header_title": "libhipcxx 3.0.2",
    "header_link": "",
    "version_list_link": "https://rocm.docs.amd.com/projects/libhipcxx/en/latest/release.html",
    "nav_secondary_items": {
        "GitHub": "https://github.com/AMD-AIOSS/libhipcxx",
        "Blogs": "https://rocm.blogs.amd.com/",
        "ROCm Developer Hub": "https://www.amd.com/en/developer/resources/rocm-hub.html",
        "Instinct™ Docs": "https://instinct.docs.amd.com/",
        "Infinity Hub": "https://www.amd.com/en/developer/resources/infinity-hub.html",
        "Support": "https://github.com/AMD-AIOSS/libhipcxx/issues",
    },
    "link_main_doc": False,
    "use_download_button": True,
    "repository_url": "https://github.com/AMD-AIOSS/libhipcxx",
    "use_repository_button": True,
    "use_issues_button": True,
    # Add any additional theme options here
}

'''
docs_header_version is used to manually configure the version in the header. If
there exists a non-null value mapped to docs_header_version, then the header in
the documentation page will contain the given version string.
'''
html_context = {"docs_header_version": "3.15"}


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

exclude_patterns = ['_includes/**']

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