#!/usr/bin/env bash
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
#
# Generate the libhipcxx documentation using the rocm-docs-core layout.
#
# This is the AMD/HIP equivalent of the upstream NVIDIA cccl docs build,
# which invoked the omniverse "repo" tool (`./repo.sh docs`) -- a build
# system we don't ship here. Instead we go straight to plain
# `sphinx-build` against the rocm-docs-core integration that conf.py
# already wires up via `from rocm_docs import ROCmDocs`.
#
# Usage:
#   ./docs/gen_docs.bash                          # build into _build/html
#   ./docs/gen_docs.bash --no-pip-install         # skip dependency install
#   ./docs/gen_docs.bash --serve                  # build + start a local
#                                                  preview server on :8000
#   ./docs/gen_docs.bash --watch                  # watch sources, rebuild
#                                                  on save + auto-reload
#                                                  browser (uses
#                                                  sphinx-autobuild)
#   ./docs/gen_docs.bash --output <dir>           # custom output directory
#   ./docs/gen_docs.bash [-- sphinx args...]      # pass extra args to
#                                                  sphinx-build (e.g.
#                                                  '-- -n' for nit-picky)
#
# Output: HTML in <output>/html (default: docs/_build/html). The
# generated `.sphinx/_toc.yml` (produced from `.sphinx/_toc.yml.in` by
# rocm-docs-core) is left next to its template for inspection.
#
# Live-preview tips inside VSCode:
#
#   * `--watch` mode pairs naturally with VSCode's Live Server or Live
#     Preview extensions: run `./gen_docs.bash --watch` in a terminal,
#     then open <output>/html/index.html in VSCode and "Go Live" -- the
#     browser pane auto-reloads on every save. (sphinx-autobuild also
#     serves its own livereload-enabled HTTP server on :8000 by
#     default, so the VSCode extension isn't strictly needed.)

set -euo pipefail

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)
cd "${SCRIPT_PATH}"

# Sphinx + rocm-docs-core require a UTF-8 locale. Provide one explicitly
# so the script works in stripped-down container environments.
export LC_ALL=${LC_ALL:-C.UTF-8}
export LANG=${LANG:-C.UTF-8}

INSTALL_DEPS=1
SERVE=0
WATCH=0
OUTPUT_DIR="_build"
SPHINX_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-pip-install)
            INSTALL_DEPS=0
            shift
            ;;
        --serve)
            SERVE=1
            shift
            ;;
        --watch)
            WATCH=1
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --)
            shift
            SPHINX_ARGS=("$@")
            break
            ;;
        -h|--help)
            sed -n 's/^# \?//p' "${BASH_SOURCE[0]}" | sed -n '/^Usage:/,/^Output:/p'
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (try --help)" >&2
            exit 2
            ;;
    esac
done

if [[ "${INSTALL_DEPS}" == "1" ]]; then
    echo "[gen_docs] installing requirements (use --no-pip-install to skip)"
    python3 -m pip install --quiet --upgrade -r requirements.txt
fi

# rocm-docs-core's util.get_branch() walks every remote / packed ref in
# the surrounding git repo to pick a "current branch" for the rocm-docs-
# core version selector. On a checkout with stale .git/config branch
# sections (e.g. an old [branch ...] block that refers to a renamed
# remote ref) that walk can raise ValueError mid-config-init and abort
# the build. Short-circuit it with ROCM_DOCS_REMOTE_DETAILS="<url>,<branch>"
# if the caller didn't already set it, decoupling the docs build from
# the user's local .git/ state.
if [[ -z "${ROCM_DOCS_REMOTE_DETAILS:-}" ]]; then
    _repo_url="$(git -C "${SCRIPT_PATH}" config --get remote.origin.url 2>/dev/null || true)"
    _branch="$(git -C "${SCRIPT_PATH}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    # Normalise ssh-style 'git@github.com:Owner/Repo.git' to the http URL
    # rocm-docs-core expects (mirrors the regex in
    # rocm_docs/util.py:get_repo_url).
    if [[ "${_repo_url}" =~ ^git@([^:]+):(.*)$ ]]; then
        _repo_url="http://${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
        _repo_url="${_repo_url%.git}"
    fi
    if [[ -n "${_repo_url}" && -n "${_branch}" ]]; then
        export ROCM_DOCS_REMOTE_DETAILS="${_repo_url},${_branch}"
        echo "[gen_docs] ROCM_DOCS_REMOTE_DETAILS=${ROCM_DOCS_REMOTE_DETAILS}"
    fi
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${WATCH}" == "1" ]]; then
    # sphinx-autobuild = sphinx-build + watchdog + a livereload-enabled
    # http.server. Re-runs the build automatically whenever any source
    # file (.rst / .md / conf.py / .sphinx/_toc.yml.in) changes and
    # pushes a websocket reload to every connected browser tab. Pairs
    # well with VSCode's Live Server / Live Preview extensions but
    # also works standalone on :8000.
    if ! python3 -c "import sphinx_autobuild" 2>/dev/null; then
        echo "[gen_docs] installing sphinx-autobuild for --watch mode"
        python3 -m pip install --quiet --upgrade sphinx-autobuild
    fi
    # NOTE(HIP/AMD): the --ignore patterns below are load-bearing for
    # avoiding an infinite rebuild loop. sphinx-autobuild watches the
    # source dir for changes; without these ignores the watcher sees
    # files THE BUILD ITSELF creates and treats them as fresh edits:
    #
    #   * "${OUTPUT_DIR}/*"        -- the build output directory.
    #     sphinx writes _build/html/* and _build/doctrees/* on every
    #     run; with only '--ignore "${OUTPUT_DIR}"' (no /*), the glob
    #     matches the directory NAME but not its contents.
    #   * ".sphinx/_toc.yml"       -- rocm-docs-core's projects.py
    #     regenerates this file from .sphinx/_toc.yml.in at every
    #     config-inited. Without ignoring it we get:
    #       (build) -> writes _toc.yml -> (watch fires) -> (build) -> ...
    #   * "**/.jupyter_cache/*"    -- myst-nb's cache directory under
    #     _build (defensive even with the _build glob above).
    #   * "*.swp" / "*~"           -- editor swap / backup files.
    #
    # Use absolute glob patterns ('/full/path/...') for the most
    # reliable matching in sphinx-autobuild's watchfiles backend.
    _abs_output="$(cd "${OUTPUT_DIR}"; pwd -P)"
    _abs_toc="${SCRIPT_PATH}/.sphinx/_toc.yml"
    echo "[gen_docs] watching . -> ${OUTPUT_DIR}/html  (Ctrl-C to stop)"
    echo "[gen_docs] preview on http://127.0.0.1:8000/"
    exec python3 -m sphinx_autobuild \
        --host 127.0.0.1 --port 8000 \
        --ignore "${_abs_output}/*" \
        --ignore "${_abs_output}" \
        --ignore "${_abs_toc}" \
        --ignore "*/.jupyter_cache/*" \
        --ignore "*.swp" --ignore "*~" \
        -b html \
        -d "${OUTPUT_DIR}/doctrees" \
        -D language=en \
        "${SPHINX_ARGS[@]}" \
        . "${OUTPUT_DIR}/html"
fi

echo "[gen_docs] building HTML -> ${OUTPUT_DIR}/html"
python3 -m sphinx \
    -T -E -W --keep-going \
    -b html \
    -d "${OUTPUT_DIR}/doctrees" \
    -D language=en \
    "${SPHINX_ARGS[@]}" \
    . "${OUTPUT_DIR}/html"

echo "[gen_docs] build succeeded; open ${OUTPUT_DIR}/html/index.html"

if [[ "${SERVE}" == "1" ]]; then
    echo "[gen_docs] serving on http://localhost:8000/  (Ctrl-C to stop)"
    cd "${OUTPUT_DIR}/html"
    python3 -m http.server 8000
fi
