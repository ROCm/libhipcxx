<!---
    MIT License

    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
-->

# Contributing to libhipcxx

We are enthusiastic about contributions to our code and documentation. Please
feel free to file issues where documentation or functionality is lacking or,
even better, volunteer to contribute to help close these gaps!

______________________________________________________________________

> **Security vulnerabilities** — do not open a public GitHub Issue. See [SECURITY.md](SECURITY.md) for the private reporting process.

______________________________________________________________________

## Developer policies

These policies apply to all forms of activity and engagement in this project.

> [!IMPORTANT]
> AMD employees must also follow the ROCm open source software
> contributing policies at http://u.amd.com/rocm-oss-policies.

### Governance

This project is covered by the
[ROCm Project Governance](https://github.com/ROCm/ROCm/blob/develop/GOVERNANCE.md),
which also defines the code of conduct.

### Licensing

Code contributions to this project are covered under the terms of the
[LICENSE.TXT](LICENSE.TXT) file. libhipcxx is derived from libcudacxx and
LLVM's libc++. [docs/license.rst](docs/license.rst) describes the upstream
license, including the LLVM exceptions, and the MIT terms for AMD
modifications.

### Communication channels

Issue tracking, project planning, and code contributions are managed in GitHub.
We use an open-source toolchain so that workflows can be easily replicated in any fork.

The public repository is [ROCm/libhipcxx](https://github.com/ROCm/libhipcxx).

______________________________________________________________________

## Development workflows

### Issue tracking

Before filing a new issue, search through
[existing issues](https://github.com/ROCm/libhipcxx/issues) to avoid duplicates.

General guidelines:

- If your issue is already listed, upvote it and add a comment with reproduction details.
- When in doubt, file a new issue — we'll mark duplicates accordingly.
- Provide as much information as possible: command output, GPU model, ROCm version, and
  OS version. This significantly reduces triage time.
- Check your issue regularly — we may ask follow-up questions.

### Code style

Follow the [ROCm/TheRock style guides](https://github.com/ROCm/TheRock/tree/main/docs/development/style_guides):

- [Bash style guide](https://github.com/ROCm/TheRock/blob/main/docs/development/style_guides/bash_style_guide.md)
- [CMake style guide](https://github.com/ROCm/TheRock/blob/main/docs/development/style_guides/cmake_style_guide.md)
- [GitHub Actions style guide](https://github.com/ROCm/TheRock/blob/main/docs/development/style_guides/github_actions_style_guide.md)
- [Python style guide](https://github.com/ROCm/TheRock/blob/main/docs/development/style_guides/python_style_guide.md)

### New feature or test development

Discussion about new features and test areas is welcome by filing a [GitHub issue](https://github.com/ROCm/libhipcxx/issues).

### Pull Requests

All contributions should be submitted through a Pull Request (PR).

Before submitting a PR:

- Ensure the change addresses an existing issue or documented requirement.
- Run all applicable tests and validate the results.
- Update documentation as needed.
- Ensure all required GitHub Actions checks pass.
- Resolve review feedback before requesting approval.

PRs require:

- Approval from the appropriate CODEOWNERS.
- Successful completion of required status checks.
- Compliance with repository security and quality requirements.

### Branching

- Do not commit directly to protected branches. Development integrates on `amd-develop`.
- Create a feature branch for all changes.
- Keep changes focused and atomic.
- Rebase or update branches regularly to minimize merge conflicts.

### Commit Messages

- Use clear and descriptive commit messages.

### Testing

Build and test instructions are in the documentation for this revision:

- [Build from source](docs/install/source-build.rst)
- [Run the tests](docs/how-to/run-libhipcxx-tests.rst)

Contributors are expected to:

- Execute relevant unit and integration tests.
- Verify that existing functionality is not impacted.
- Include test coverage for new functionality when applicable.
- Document any known limitations or test exclusions.

### Documentation

Changes that impact functionality, configuration, APIs, or user workflows should include corresponding documentation updates. The documentation source is the `docs` folder. See the [README](README.md) for the published site and the current in-repo pages.

### Security Requirements

Contributors must not:

- Commit secrets, tokens, passwords, or credentials.
- Introduce vulnerable dependencies without justification.
- Bypass security controls or required security reviews.

All contributions may be subject to automated security scanning.
