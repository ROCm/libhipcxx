# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# requires: pip install -r requirements.txt
#   (rocm-docs-core pulls sphinx, sphinx-external-toc, and the rocm-docs
#    theme; myst-parser is needed for the .md docs in this tree.)

from rocm_docs import ROCmDocs


# -- Project information -----------------------------------------------------
project = "libhipcxx"
author = "Advanced Micro Devices, Inc. <libhipcxx.maintainer@amd.com>"
copyright = "Copyright (c) 2024-2026 Advanced Micro Devices, Inc."
os_support = ["linux"]
date = "2026-05-18"

default_role = "py:obj"  # so `foo` expands to :py:obj:`foo`

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "special-members": "__init__, __getitem__",
    "inherited-members": True,
    "show-inheritance": True,
    "imported-members": False,
}

# -- rocm-docs-core integration ----------------------------------------------
# libhipcxx isn't yet a registered project in the rocm-docs-core projects
# registry (rocm_docs/data/projects.yaml). Until it is, leave the
# 'external_projects_current_project' default and don't claim to be 'hip'
# (we are HIP-adjacent but not HIP itself, and posing as 'hip' would put
# our build under the HIP intersphinx target which isn't right).
external_projects_remote_repository = ""

# Disable the bulk intersphinx fetching. The rocm-docs-core default
# ('external_projects = "all"') would load intersphinx inventories from
# every project in rocm_docs/data/projects.yaml (~95 entries). Several
# of those URLs serve 404s (instinct.docs.amd.com/objects.inv,
# rocm-llmext-internal, rocm-ls-internal, ...), and sphinx >= 9.x
# treats an empty-body inventory as a hard ExtensionError that aborts
# the build rather than a recoverable warning. libhipcxx doesn't
# cross-reference any of those external projects from its prose, so
# empty-list the projects to opt out.
external_projects = []

# The docs are standalone: ``docs/index.rst`` is a rocm-docs-core style
# landing page with grid cards, and the prose that used to be pulled out
# of the README's tagged blocks now lives in ``docs/conceptual/``,
# ``docs/install/``, and ``docs/reference/``. If the article_pages
# metadata is needed in the future (e.g. for an explicit "blog-style"
# landing page), add an entry for the root ``index``.
article_pages = []

docs_core = ROCmDocs(project)
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

extensions += [
    "sphinx.ext.autodoc",  # autodoc from Python docstrings (for any
                           # future Python bindings)
]

# -- Excluded source files ---------------------------------------------------
# Files that exist on disk for legacy / upstream-mirror reasons but are not
# wired into the rocm-docs-core ToC and would otherwise warn as
# [toc.not_included]. The contents are NVIDIA-cccl multi-project boilerplate
# (cpp.rst / python.rst link to upstream nvidia.github.io URLs we don't
# host) or trivial version markers (VERSION.md).
exclude_patterns = [
    "VERSION.md",
    "cpp.rst",
    "python.rst",
    "_build",
    "_repo",
    "Thumbs.db",
    ".DS_Store",
]

# -- Warning suppression -----------------------------------------------------
# Suppress two categories of warnings that are noise in this tree:
#
# * 'etoc.toctree': sphinx-external-toc complains about every ``toctree::``
#   directive it finds inside our .rst files because the ``.sphinx/_toc.yml``
#   is the single source of top-level navigation. Per-page toctrees in the
#   .rst files are still useful for in-page section indexes (they render
#   the local subtree as a list), so we keep them and silence this warning
#   category instead of stripping them out.
#
# * 'intersphinx.external': bulk-loading external intersphinx inventories
#   is disabled via 'external_projects = []' above (see rationale there),
#   but we keep this category suppressed defensively in case any specific
#   intersphinx URL is re-enabled later and fails to resolve in an offline
#   environment.
suppress_warnings = [
    "etoc.toctree",
    "intersphinx.external",
]

# Suppress the single remaining rocm-docs-core warning, which is emitted
# unconditionally for projects that aren't in rocm_docs/data/projects.yaml:
#
#   "Current project 'libhipcxx' not found in projects.
#    Did you forget to set 'external_projects_current_project' to the name
#    of the current project?"
#
# We cannot suppress it via 'suppress_warnings' because it's emitted
# unconditionally from rocm_docs.projects._get_current_project as a
# generic logger.warning() rather than via a sphinx warning category.
# The only ways out are (a) upstream a 'libhipcxx:' entry to rocm-docs-
# core's project registry, or (b) install a logging filter that drops
# this one specific message. (b) here.
import logging  # noqa: E402

class _RocmDocsUnknownProjectFilter(logging.Filter):
    def filter(self, record):  # type: ignore[no-untyped-def]
        return "not found in projects" not in record.getMessage()

for _logger_name in ("sphinx.rocm_docs.projects", "rocm_docs.projects"):
    logging.getLogger(_logger_name).addFilter(
        _RocmDocsUnknownProjectFilter()
    )
