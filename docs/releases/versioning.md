<!-- MIT License
  -- 
  -- Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
  -- 
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  -- 
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  -- 
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
  -->

# Versioning

libhipcxx is versioned along two axes:

- API Version: A 3-component semantic version for the programmatic interface.
    - Follows [semantic versioning].
    - A single API version is supported in a given snapshot of the codebase.
    - An API version is released when completed.
        The name of the release is the API version.
- ABI Version: An integer version for the binary interface.
    - Multiple ABI versions may be supported in a given snapshot of the codebase.
    - New ABI versions may be introduced at any time.
        The latest available ABI version is always the default.

## Application Programming Interface (API)

### What is API?

The Application Programming Interface (API) of a software library is programming
  model and set of entities, represented by programming language constructs,
  that are intentionally exposed to provide the functionality of the library.

A library's API includes, but is not limited to:

- The structure and filenames of headers intended for direct inclusion in user
      code.
- The namespaces intended for direct use in user code.
- The declarations and/or definitions of functions, classes, and variables
      located in headers and intended for direct use in user code.
- The semantics of functions, classes, and variables intended for direct use in
      user code.
- Any identifiers intended to be found by [Argument Dependent Lookup],
      regardless of which namespaces they are in.

### libhip++ API Versioning

The libhipcxx API uses [semantic versioning].
The versioning scheme, `MMM.mmm.ppp`, consists of three components and four
  macros defined in `<cuda//std/version>`:

- `MMM`/`_LIBHIPCXX_HIP_API_VERSION_MAJOR`: Major version, an 8 bit unsigned
      integer.
    The major version represents API stability to the best of our ability
        (outstanding, but not perfect).
    When API-backwards-incompatible changes are made, this component is
        incremented.
    ABI changes that do not have an associated API-backwards-incompatible change
        do not trigger a new major release.
- `mmm`/`_LIBHIPCXX_HIP_API_VERSION_MINOR`: Minor version, an 8 bit unsigned
        integer.
    When API-backwards-compatible features are added are made, this component is
        incremented.
    Such changes may be made at any time.
- `ppp`/`_LIBHIPCXX_HIP_API_VERSION_PATCH`: Subminor version, an 8 bit
        unsigned integer.
    When changes are made that do not qualify for an increase in either of the
        other two versions, it is incremented.
    Such changes may be made at any time.
- `MMMmmmppp`/`_LIBHIPCXX_HIP_API_VERSION`: A concatenation of the decimal
    digits of all three components, a 32 bit unsigned integer.

A single API version is supported in any given snapshot of the codebase.
Only the latest API version is supported and maintained.

When work completes on an API version on the main development branch, a
  release is snapped.
The name of that release is the API version.
After a release is snapped, the API version on the main development branch
  is incremented and work begins on the next release.

## Application Binary Interface (ABI)

### What is ABI?

The Application Binary Interface (ABI) of a software library is the convention
  for how:

- The library's entities are represented in machine code, and
- Library entities built in one translation unit interact with entities from
    another.

A library's ABI includes, but is not limited to:

- The mangled names of functions.
- The mangled names of types, including instantiations of class templates.
- The number of bytes (`sizeof`) and the alignment of objects and types.
- The semantics of the bytes in the binary representation of an object.
- The register-level calling conventions governing parameter passing and
      function invocation.

Parts of this section were based on
  [P2028R0: What is ABI, and what Should Standard C++ Do About It?].

### libhip++ ABI Versioning

The libhipcxx ABI version scheme is a single 16 bit unsigned
  integer value.
An ABI version represents stability in the ABI to the best of our ability
  (excellent, but not perfect).
When any changes are made to the ABI, even seemingly minor ones, the ABI
  version is incremented.
libhipcxx does not maintain long-term ABI stability.
Promising long-term ABI stability would prevent us from fixing mistakes and
  providing best in class performance.
So, we make no such promises.

A snapshot of the codebase may support multiple ABI versions at the same time.
They always use the latest available ABI version by default.
The macro `_LIBHIPCXX_HIP_ABI_VERSION_LATEST` from `<cuda//std/version>`
  defines the value of the latest ABI version.

New ABI versions may be introduced at any point in time, which means that the
  default ABI version may change in any release.
A subset of older ABI versions can be used instead by defining
  `_LIBHIPCXX_HIP_ABI_VERSION` to the desired version.

A program is ill-formed, no diagnostic required, if it uses two different
  translation units compiled with a different libhipcxx ABI
  version.
For example, all of the following is disallowed:

- Compiling `foo.cu` with ABI version 1, `bar.cu` with ABI version 2, and
      linking `foo.cu` and `bar.cu` into one program.
- Compiling `foo.cu` with ABI version 2 and linking it with a library,
      `libbar.so`, compiled with ABI version 1.
- Compiling `foo.cu` with ABI version 1 and linking it with a library,
      `libbar.a`, compiled with ABI version 2.

Every namespace used by the libhipcxx has an
  [inline namespace] `__N` (where `N` is the ABI version); this is known as an
  ABI namespace.
Nested namespaces have their own ABI namespace, e.g. `cuda::__N::` and
  `cuda::stdd::__N::`.
Some library facilities which are tightly coupled with the C++ compiler, such as
  `std::initializer_list`, are not in the ABI namespace.
The ABI namespace will be rarely be encountered, because, in most
  circumstances, all the members of an inline namespace are introduced into the
  enclosing namespace.
So, `cuda::stdd::__N::atomic` is available as cuda::stdtd::atomic`.

When including libhipcxx in a translation unit, a single
  ABI version and single ABI namespace will be defined.
You cannot utilize two different ABI versions in a single translation unit by
   explicitly using ABI namespaces.

ABI namespaces aid in diagnosing the use of translation units compiled with
  different ABI versions.
We want to try and help anyone who is mixing code from different ABIs by loudly
  breaking them at compile time instead of quietly failing them at runtime.
Suppose we have one translation unit that uses ABI version 3 and defines a
  function `void negate(cuda::atomic<float>&)`.
`cuda::atomic<float>` is just another name for `cuda::__3::atomic<float>`, so
  `negate`'s signature is really `void negate(cuda::__3::atomic<float>&)`.
If we use `negate` in another translation unit which was compiled with ABI
  version 4, we will get an error when we try to link these two translation
  units together.
The second translation unit is looking for a function
  `void negate(cuda::__3::atomic<float>&)`, which does not exist.

However, we must be careful, because ABI namespaces cannot diagnosis all mixing
  of different ABI versions.
Let's say we a translation unit compiled with ABI version 3 that contains this
  code:

```hip
struct sum { cuda::atomic<float> };
void negate(sum&);
```

If we try to use `negate` in a second translation unit compiled with
  ABI version 4, our program will be ill-formed but we will not get a link-time
  failure, because while `cuda::atomic` is in an ABI namespace, `sum` is not.
`cuda::atomic`'s size and layout may have changed across ABI versions, so the
  size and layout of `sum` may be different in each translation unit.
Our program has violated the [One Definition Rule] and thus has undefined
  behavior.
It will likely fail in unpredictable ways at run-time.

If ABI stability is critical to us, we could explicitly use
  `cuda::__3::atomic<float>` in `sum`.
Then, we would get a compile-time error when attempting to compile the second
  translation unit using ABI version 4, because `cuda::__3::atomic<float>` would
  not be defined.
However, this is generally not recommended because it has the significant
  downside that our code does not automatically migrate to newer ABI versions.

We recommend that you always recompile your code and dependencies with the
  latest ROCm version and use the latest libhipcxx ABI.
[Live at head].

## `experimental` Namespaces

Some libhipcxx APIs live in a nested `experimental`
  namespace.
We make absolutely no guarantees about such features.
Their API and ABI is subject to change or wholesale removal at any time
  without any notice.

## Deprecation

Prior to making substantial API-backwards-incompatible changes, making ABI
  changes, or modifying the platforms and compilers we support, we will
  typically notify users by deprecating the things that are changing.
Deprecations will come in the form of programmatic warnings which can be
  disabled with a macro.
The deprecation period will depend on the impact of the change, but will usually
  last three to six months.


[releases section]: ../releases.md
[changelog]: changelog.md

[semantic versioning]: https://semver.org
[live at head]: https://www.youtube.com/watch?v=tISy7EJQPzI&t=1032s

[P2028R0: What is ABI, and what Should Standard C++ Do About It?]: https://wg21.link/P2028R0

[Argument Dependent Lookup]: https://en.cppreference.com/w/cpp/language/adl
[One Definition Rule]: https://en.cppreference.com/w/cpp/language/definition#One_Definition_Rule
[inline namespace]: https://en.cppreference.com/w/cpp/language/namespace#Inline_namespaces


