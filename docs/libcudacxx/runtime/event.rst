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
  :description: API reference for cuda::event_ref, cuda::event and cuda::timed_event, the wrappers around a HIP event in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, event, event_ref, timed_event, hipEvent_t, synchronization

.. _cccl-runtime-event:

Events
======

This page documents ``cuda::event_ref``, ``cuda::event`` and ``cuda::timed_event``, which wrap a ``hipEvent_t``.

Event is a snapshot of execution state of a stream. It can be used to synchronize work submitted to a stream up to a certain point, establish dependency between streams or measure time passed between two events.
See `Events for synchronization <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html#events-for-synchronization>`__ for an introduction to HIP events.

:cpp:class:`cuda::event_ref`
--------------------------------------------------
.. _cccl-runtime-event-event-ref:

:cpp:class:`cuda::event_ref` is a non-owning wrapper around a ``hipEvent_t``. It prevents unsafe implicit constructions from
``nullptr`` or integer literals and provides convenient helpers:

- ``record(cuda::stream_ref)``: record the event on a stream
- ``sync()``: wait for the recorded work to complete
- ``is_done()``: non-blocking completion query
- comparison operators against other :cpp:class:`cuda::event_ref` or ``hipEvent_t``

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/stream>

   void record_on_stream(cuda::stream_ref stream, hipEvent_t raw_handle) {
     cuda::event_ref e{raw_handle};
     e.record(stream);
   }

:cpp:class:`cuda::event`
--------------------------------------------
.. _cccl-runtime-event-event:

:cpp:class:`cuda::event` is an owning wrapper around a ``hipEvent_t`` (with timing disabled). It inherits from
:cpp:class:`cuda::event_ref` and provides all of its functionality. It also creates and destroys the native event, can be moved (but
not copied), and can release ownership via ``release()``. Construction can target a specific :cpp:class:`cuda::device_ref`
or record immediately on a :cpp:class:`cuda::stream_ref`.

Availability: libhipcxx 3.4

.. code:: cpp

   #include <cuda/stream>
   #include <cuda/devices>
   #include <cuda/std/optional>

   cuda::std::optional<cuda::event> query_and_record_on_stream(cuda::stream_ref stream) {
     if (stream.is_done()) {
       return cuda::std::nullopt;
     } else {
       return cuda::event{stream};
     }
   }

.. _cccl-runtime-event-timed-event:

:cpp:class:`cuda::timed_event`
-----------------------------------------------------

:cpp:class:`cuda::timed_event` is an owning wrapper for a timed ``hipEvent_t``. It inherits from :cpp:class:`cuda::event` and provides
all of its functionality.
It also supports elapsed-time queries between two events via ``operator-``, returning
:cpp:class:`cuda::std::chrono::nanoseconds`.

Availability: libhipcxx 3.4

.. code:: cpp

   #include <cuda/stream>
   #include <cuda/std/chrono>

   template <typename F>
   cuda::std::chrono::nanoseconds measure_execution_time(cuda::stream_ref stream, F&& f) {
     cuda::timed_event start{stream};
     f(stream);
     cuda::timed_event end{stream};
     return end - start;
   }
