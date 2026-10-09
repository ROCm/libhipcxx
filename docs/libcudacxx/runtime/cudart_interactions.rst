..
    MIT License

    Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

.. meta::
  :description: How the libhipcxx Runtime API types interoperate with native HIP runtime handles, device selection and the default stream in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, runtime interop, hipStream_t, native handle, device selection, default stream

.. _cccl-runtime-cudart-interactions:

HIP runtime interactions
========================

This page documents how the libhipcxx runtime types interoperate with existing HIP runtime code.

Some runtime objects have a non-owning ``_ref`` counterpart (for example, :cpp:struct:`cuda::stream` and
:cpp:class:`cuda::stream_ref`). Prefer the
owning type for lifetime management, and use the ``_ref`` type for code that would otherwise accept a C++ reference but
needs to interoperate with existing HIP runtime code.

libhipcxx runtime types that wrap HIP runtime handles support interoperating with HIP runtime handles via ``get()``,
constructors that accept native handles, ``release()``, and ``from_native_handle`` helpers. This makes it straightforward
to bridge between libhipcxx runtime APIs and existing HIP runtime code without losing ownership clarity.

Use ``get()`` on both owning and non-owning types. Constructors from native handles are intended for ``_ref`` wrappers,
while ``release()`` and ``from_native_handle`` are for owning objects that transfer or assume ownership.

Example: handle interop patterns
--------------------------------

.. code:: cpp

   #include <cuda/stream>

   void use_handle_interop(cuda::device_ref device, hipStream_t raw_stream) {
     // _ref from native handle (non-owning).
     cuda::stream_ref borrowed{raw_stream};

     // Universal handle access.
     assert(borrowed.get() == raw_stream);

     // Owning from native handle (assumes ownership).
     auto owned = cuda::stream::from_native_handle(raw_stream);

     assert(owned.get() == raw_stream);

     // Release ownership back to the HIP runtime.
     hipStream_t released = owned.release();

     assert(released == raw_stream);
   }

Device selection
----------------

The Runtime API emphasizes explicit device selection. Most entry points take a :cpp:class:`cuda::device_ref` or a
device-bound resource (such as :cpp:struct:`cuda::stream`) rather than relying on implicit global state like
``hipSetDevice``. This
makes device ownership and lifetime clearer, especially in multi-GPU code.

The current device can still be set via the HIP runtime, but libhipcxx runtime APIs ignore that global state and require an
explicit device argument. The libhipcxx runtime also does not provide APIs that read or mutate the current device, by design.
When a libhipcxx runtime API needs a specific device to be current, it selects that device with ``hipSetDevice`` and
restores the previously current device (as returned by ``hipGetDevice``) before returning.


Default stream interop
----------------------

The HIP default (NULL) stream is not exposed as a first-class runtime object because it is tied to implicit per-device
state and encourages hidden dependencies. Instead, it can be wrapped into :cpp:class:`cuda::stream_ref` when needed for
interop.

.. note::

   When wrapping the NULL stream, the current device must be set explicitly first. HIP binds the NULL stream to the
   active device, so the wrapper must be created after selecting the correct device.

Example: wrapping the default stream
------------------------------------

.. code:: cpp

   #include <cuda/stream>

   void use_default_stream(int device_id) {
     hipSetDevice(device_id);

     cuda::stream_ref default_stream{hipStreamPerThread};
     // Use default_stream with libhipcxx runtime APIs.
   }

..
   libhipcxx does not use driver API contexts on HIP; the current device is managed through hipSetDevice.

   The above applies to Driver API interop cases as well, where the current context must be managed by the user rather than
   the current device setting.
