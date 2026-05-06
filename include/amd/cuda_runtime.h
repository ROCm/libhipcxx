// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#if !defined(__HIPCC_RTC__)
#  include <hip/hip_runtime.h>
#endif // !__HIPCC_RTC__
// NOTE(HIP/AMD): Under HIPRTC, the HIPRTC-specific runtime stub definitions
// (hipPointerAttribute_t, hipMemoryType, hipError_t, etc.) are provided by the
// test infrastructure header (test/utils/amd/hiprtc/hiprtcc_common.h) which
// prepends them to every compiled source.
#include <amd/amd_utils.h>

#define CUDART_VERSION 0

// types
#ifndef cudaDeviceProp
#define cudaDeviceProp hipDeviceProp_t
#endif
#ifndef cudaError_t
#  define cudaError_t hipError_t
#endif
#ifndef cudaDeviceAttr
#  define cudaDeviceAttr hipDeviceAttribute_t
#endif
#ifndef cudaComputeMode
#  define cudaComputeMode hipComputeMode
#endif
#ifndef cudaComputeModeDefault
#  define cudaComputeModeDefault hipComputeModeDefault
#endif
#ifndef cudaComputeModeExclusive
#  define cudaComputeModeExclusive hipComputeModeExclusive
#endif
#ifndef cudaComputeModeProhibited
#  define cudaComputeModeProhibited hipComputeModeProhibited
#endif
#ifndef cudaComputeModeExclusiveProcess
#  define cudaComputeModeExclusiveProcess hipComputeModeExclusiveProcess
#endif
#ifndef cudaEvent_t
#  define cudaEvent_t hipEvent_t
#endif
#ifndef cudaMemPool_t
#  define cudaMemPool_t hipMemPool_t
#endif
#ifndef cudaStream_t
#  define cudaStream_t hipStream_t
#endif
#ifndef cudaMemPoolAttr
#  define cudaMemPoolAttr hipMemPoolAttr
#endif
#ifndef cudaMemPoolProps
#  define cudaMemPoolProps hipMemPoolProps
#endif
#ifndef cudaMemAllocationHandleType
#  define cudaMemAllocationHandleType hipMemAllocationHandleType
#endif
#ifndef cudaPointerAttributes
#  define cudaPointerAttributes hipPointerAttribute_t
#endif
#ifndef cudaAccessProperty
#  define cudaAccessProperty hipAccessProperty
#endif
#ifndef cudaAccessPropertyNormal
#  define cudaAccessPropertyNormal hipAccessPropertyNormal
#endif
#ifndef cudaAccessPropertyStreaming
#  define cudaAccessPropertyStreaming hipAccessPropertyStreaming
#endif
#ifndef cudaAccessPropertyPersisting
#  define cudaAccessPropertyPersisting hipAccessPropertyPersisting
#endif
// macros, enum constant definitions
// NOTE: C++ `constexpr` might cause redefinition errors while #define only results in a warning in this case.
//       Such redefinitions might happen when a code includes multiple "reverse hipfication" header files
//       like this one from a number of other projects. Therefore, we prefer to use #define.
#ifndef cudaStreamLegacy
#  define cudaStreamLegacy ((hipStream_t) nullptr)
#endif
#ifndef cudaStreamPerThread
#  define cudaStreamPerThread hipStreamPerThread
#endif
#ifndef cudaMemcpy
#  define cudaMemcpy hipMemcpy
#endif
#ifndef cudaMemcpyToSymbol
#  define cudaMemcpyToSymbol hipMemcpyToSymbol
#endif
#ifndef cudaMemcpyDefault
#  define cudaMemcpyDefault hipMemcpyDefault
#endif
#ifndef cudaMemcpyDeviceToHost 
#  define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost 
#endif
#ifndef cudaMemcpyHostToDevice 
#  define cudaMemcpyHostToDevice hipMemcpyHostToDevice 
#endif
#ifndef cudaMemPoolAttrReleaseThreshold
#  define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#endif
#ifndef cudaDevAttrMemoryPoolSupportedHandleTypes
#  define cudaDevAttrMemoryPoolSupportedHandleTypes hipDeviceAttributeMemoryPoolSupportedHandleTypes
#endif
#ifndef cudaDevAttrMemoryPoolsSupported
#  define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported
#endif
#ifndef cudaDevAttrL2CacheSize
#  define cudaDevAttrL2CacheSize hipDeviceAttributeL2CacheSize
#endif
#ifndef cudaDevAttrConcurrentManagedAccess
#define cudaDevAttrConcurrentManagedAccess hipDeviceAttributeConcurrentManagedAccess
#endif
#ifndef cudaDevAttrClockRate
#  define cudaDevAttrClockRate hipDeviceAttributeClockRate
#endif
#ifndef cudaDevAttrGpuOverlap
#  define cudaDevAttrGpuOverlap hipDeviceAttributeAsyncEngineCount
#endif
#ifndef cudaDevAttrCanMapHostMemory
#  define cudaDevAttrCanMapHostMemory hipDeviceAttributeCanMapHostMemory
#endif
#ifndef cudaDevAttrComputeMode
#  define cudaDevAttrComputeMode hipDeviceAttributeComputeMode
#endif
#ifndef cudaDevAttrConcurrentKernels
#  define cudaDevAttrConcurrentKernels hipDeviceAttributeConcurrentKernels
#endif
#ifndef cudaDevAttrEccEnabled
#  define cudaDevAttrEccEnabled hipDeviceAttributeEccEnabled
#endif
#ifndef cudaDevAttrGlobalMemoryBusWidth
#  define cudaDevAttrGlobalMemoryBusWidth hipDeviceAttributeMemoryBusWidth
#endif
#ifndef cudaDevAttrIntegrated
#  define cudaDevAttrIntegrated hipDeviceAttributeIntegrated
#endif
#ifndef cudaDevAttrKernelExecTimeout
#  define cudaDevAttrKernelExecTimeout hipDeviceAttributeKernelExecTimeout
#endif
#ifndef cudaDevAttrMemoryClockRate
#  define cudaDevAttrMemoryClockRate hipDeviceAttributeMemoryClockRate
#endif
#ifndef cudaDevAttrTccDriver
#  define cudaDevAttrTccDriver hipDeviceAttributeTccDriver
#endif
#ifndef cudaDevAttrUnifiedAddressing
#  define cudaDevAttrUnifiedAddressing hipDeviceAttributeUnifiedAddressing
#endif
#ifndef cudaDevAttrStreamPrioritiesSupported
#  define cudaDevAttrStreamPrioritiesSupported hipDeviceAttributeStreamPrioritiesSupported
#endif
#ifndef cudaDevAttrGlobalL1CacheSupported
#  define cudaDevAttrGlobalL1CacheSupported hipDeviceAttributeGlobalL1CacheSupported
#endif
#ifndef cudaDevAttrLocalL1CacheSupported
#  define cudaDevAttrLocalL1CacheSupported hipDeviceAttributeLocalL1CacheSupported
#endif
#ifndef cudaDevAttrIsMultiGpuBoard
#  define cudaDevAttrIsMultiGpuBoard hipDeviceAttributeIsMultiGpuBoard
#endif
#ifndef cudaDevAttrCanFlushRemoteWrites
#  define cudaDevAttrCanFlushRemoteWrites hipDeviceAttributeHdpMemFlushCntl
#endif
#ifndef cudaDevAttrCanUseHostPointerForRegisteredMem
#  define cudaDevAttrCanUseHostPointerForRegisteredMem hipDeviceAttributeCanUseHostPointerForRegisteredMem
#endif
#ifndef cudaDevAttrComputePreemptionSupported
#  define cudaDevAttrComputePreemptionSupported hipDeviceAttributeComputePreemptionSupported
#endif
#ifndef cudaDevAttrCooperativeLaunch
#  define cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#endif
#ifndef cudaDevAttrDirectManagedMemAccessFromHost
#  define cudaDevAttrDirectManagedMemAccessFromHost hipDeviceAttributeDirectManagedMemAccessFromHost
#endif
#ifndef cudaDevAttrHostNativeAtomicSupported
#  define cudaDevAttrHostNativeAtomicSupported hipDeviceAttributeHostNativeAtomicSupported
#endif
#ifndef cudaDevAttrHostRegisterSupported
#  define cudaDevAttrHostRegisterSupported hipDeviceAttributeHostRegisterSupported
#endif
#ifndef cudaDevAttrPageableMemoryAccess
#  define cudaDevAttrPageableMemoryAccess hipDeviceAttributePageableMemoryAccess
#endif
#ifndef cudaDevAttrDeferredMappingCudaArraySupported
#  define cudaDevAttrDeferredMappingCudaArraySupported hipDeviceAttributeImageSupport
#endif
#ifndef cudaDevAttrGPUDirectRDMAFlushWritesOptions
#  define cudaDevAttrGPUDirectRDMAFlushWritesOptions hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc
#endif
#ifndef cudaDevAttrGPUDirectRDMASupported
#  define cudaDevAttrGPUDirectRDMASupported hipDeviceAttributeIsLargeBar
#endif
#ifndef cudaDevAttrHostRegisterReadOnlySupported
#  define cudaDevAttrHostRegisterReadOnlySupported hipDeviceAttributeAsicRevision
#endif
#ifndef cudaDevAttrIpcEventSupport
#  define cudaDevAttrIpcEventSupport hipDeviceAttributeFineGrainSupport
#endif
#ifndef cudaDevAttrPageableMemoryAccessUsesHostPageTables
#  define cudaDevAttrPageableMemoryAccessUsesHostPageTables hipDeviceAttributeCanUseStreamWaitValue
#endif
#ifndef cudaDevAttrSparseCudaArraySupported
#  define cudaDevAttrSparseCudaArraySupported hipDeviceAttributeVirtualMemoryManagementSupported
#endif
#ifndef cudaDevAttrManagedMemory
#define cudaDevAttrManagedMemory hipDeviceAttributeManagedMemory
#endif
#ifndef cudaDevAttrGPUDirectRDMAWritesOrdering
#  define cudaDevAttrGPUDirectRDMAWritesOrdering hipDeviceAttributeHdpRegFlushCntl
#endif
#ifndef cudaDevAttrMaxThreadsPerBlock
#  define cudaDevAttrMaxThreadsPerBlock hipDeviceAttributeMaxThreadsPerBlock
#endif
#ifndef cudaDevAttrMaxBlockDimX
#  define cudaDevAttrMaxBlockDimX hipDeviceAttributeMaxBlockDimX
#endif
#ifndef cudaDevAttrMaxBlockDimY
#  define cudaDevAttrMaxBlockDimY hipDeviceAttributeMaxBlockDimY
#endif
#ifndef cudaDevAttrMaxBlockDimZ
#  define cudaDevAttrMaxBlockDimZ hipDeviceAttributeMaxBlockDimZ
#endif
#ifndef cudaDevAttrMaxGridDimX
#  define cudaDevAttrMaxGridDimX hipDeviceAttributeMaxGridDimX
#endif
#ifndef cudaDevAttrMaxGridDimY
#  define cudaDevAttrMaxGridDimY hipDeviceAttributeMaxGridDimY
#endif
#ifndef cudaDevAttrMaxGridDimZ
#  define cudaDevAttrMaxGridDimZ hipDeviceAttributeMaxGridDimZ
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerBlock
#  define cudaDevAttrMaxSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#endif
#ifndef cudaDevAttrTotalConstantMemory
#  define cudaDevAttrTotalConstantMemory hipDeviceAttributeTotalConstantMemory
#endif
#ifndef cudaDevAttrWarpSize
#  define cudaDevAttrWarpSize hipDeviceAttributeWarpSize
#endif
#ifndef cudaDevAttrMaxPitch
#  define cudaDevAttrMaxPitch hipDeviceAttributeMaxPitch
#endif
#ifndef cudaDevAttrMaxTexture1DWidth
#  define cudaDevAttrMaxTexture1DWidth hipDeviceAttributeMaxTexture1DWidth
#endif
#ifndef cudaDevAttrMaxTexture1DLinearWidth
#  define cudaDevAttrMaxTexture1DLinearWidth hipDeviceAttributeMaxTexture1DLinear
#endif
#ifndef cudaDevAttrMaxTexture1DMipmappedWidth
#  define cudaDevAttrMaxTexture1DMipmappedWidth hipDeviceAttributeMaxTexture1DMipmap
#endif
#ifndef cudaDevAttrMaxTexture2DWidth
#  define cudaDevAttrMaxTexture2DWidth hipDeviceAttributeMaxTexture2DWidth
#endif
#ifndef cudaDevAttrMaxTexture2DHeight
#  define cudaDevAttrMaxTexture2DHeight hipDeviceAttributeMaxTexture2DHeight
#endif
#ifndef cudaDevAttrMaxTexture2DLinearWidth
#  define cudaDevAttrMaxTexture2DLinearWidth hipDeviceAttributeMaxTexture2DLinear
#endif
#ifndef cudaDevAttrMaxTexture2DLinearHeight
#  define cudaDevAttrMaxTexture2DLinearHeight hipDeviceAttributeMaxTexture2DLinear
#endif
#ifndef cudaDevAttrMaxTexture2DLinearPitch
#  define cudaDevAttrMaxTexture2DLinearPitch hipDeviceAttributeMaxTexture2DLinear
#endif
#ifndef cudaDevAttrMaxTexture2DMipmappedWidth
#  define cudaDevAttrMaxTexture2DMipmappedWidth hipDeviceAttributeMaxTexture2DMipmap
#endif
#ifndef cudaDevAttrMaxTexture2DMipmappedHeight
#  define cudaDevAttrMaxTexture2DMipmappedHeight hipDeviceAttributeMaxTexture2DMipmap
#endif
#ifndef cudaDevAttrMaxTexture3DWidth
#  define cudaDevAttrMaxTexture3DWidth hipDeviceAttributeMaxTexture3DWidth
#endif
#ifndef cudaDevAttrMaxTexture3DHeight
#  define cudaDevAttrMaxTexture3DHeight hipDeviceAttributeMaxTexture3DHeight
#endif
#ifndef cudaDevAttrMaxTexture3DDepth
#  define cudaDevAttrMaxTexture3DDepth hipDeviceAttributeMaxTexture3DDepth
#endif
#ifndef cudaDevAttrMaxTexture3DWidthAlt
#  define cudaDevAttrMaxTexture3DWidthAlt hipDeviceAttributeMaxTexture3DAlt
#endif
#ifndef cudaDevAttrMaxTexture3DHeightAlt
#  define cudaDevAttrMaxTexture3DHeightAlt hipDeviceAttributeMaxTexture3DAlt
#endif
#ifndef cudaDevAttrMaxTexture3DDepthAlt
#  define cudaDevAttrMaxTexture3DDepthAlt hipDeviceAttributeMaxTexture3DAlt
#endif
#ifndef cudaDevAttrMaxTextureCubemapWidth
#  define cudaDevAttrMaxTextureCubemapWidth hipDeviceAttributeMaxTextureCubemap
#endif
#ifndef cudaDevAttrMaxTexture1DLayeredWidth
#  define cudaDevAttrMaxTexture1DLayeredWidth hipDeviceAttributeMaxTexture1DLayered
#endif
#ifndef cudaDevAttrMaxTexture1DLayeredLayers
#  define cudaDevAttrMaxTexture1DLayeredLayers hipDeviceAttributeMaxTexture1DLayered
#endif
#ifndef cudaDevAttrMaxTexture2DLayeredWidth
#  define cudaDevAttrMaxTexture2DLayeredWidth hipDeviceAttributeMaxTexture2DLayered
#endif
#ifndef cudaDevAttrMaxTexture2DLayeredHeight
#  define cudaDevAttrMaxTexture2DLayeredHeight hipDeviceAttributeMaxTexture2DLayered
#endif
#ifndef cudaDevAttrMaxTexture2DLayeredLayers
#  define cudaDevAttrMaxTexture2DLayeredLayers hipDeviceAttributeMaxTexture2DLayered
#endif
#ifndef cudaDevAttrMaxTextureCubemapLayeredWidth
#  define cudaDevAttrMaxTextureCubemapLayeredWidth hipDeviceAttributeMaxTextureCubemapLayered
#endif
#ifndef cudaDevAttrMaxTextureCubemapLayeredLayers
#  define cudaDevAttrMaxTextureCubemapLayeredLayers hipDeviceAttributeMaxTextureCubemapLayered
#endif
#ifndef cudaDevAttrMaxSurface1DWidth
#  define cudaDevAttrMaxSurface1DWidth hipDeviceAttributeMaxSurface1D
#endif
#ifndef cudaDevAttrMaxSurface2DWidth
#  define cudaDevAttrMaxSurface2DWidth hipDeviceAttributeMaxSurface2D
#endif
#ifndef cudaDevAttrMaxSurface2DHeight
#  define cudaDevAttrMaxSurface2DHeight hipDeviceAttributeMaxSurface2D
#endif
#ifndef cudaDevAttrMaxSurface3DWidth
#  define cudaDevAttrMaxSurface3DWidth hipDeviceAttributeMaxSurface3D
#endif
#ifndef cudaDevAttrMaxSurface3DHeight
#  define cudaDevAttrMaxSurface3DHeight hipDeviceAttributeMaxSurface3D
#endif
#ifndef cudaDevAttrMaxSurface3DDepth
#  define cudaDevAttrMaxSurface3DDepth hipDeviceAttributeMaxSurface3D
#endif
#ifndef cudaDevAttrMaxSurface1DLayeredWidth
#  define cudaDevAttrMaxSurface1DLayeredWidth hipDeviceAttributeMaxSurface1DLayered
#endif
#ifndef cudaDevAttrMaxSurface1DLayeredLayers
#  define cudaDevAttrMaxSurface1DLayeredLayers hipDeviceAttributeMaxSurface1DLayered
#endif
#ifndef cudaDevAttrMaxSurface2DLayeredWidth
#  define cudaDevAttrMaxSurface2DLayeredWidth hipDeviceAttributeMaxSurface2DLayered
#endif
#ifndef cudaDevAttrMaxSurface2DLayeredHeight
#  define cudaDevAttrMaxSurface2DLayeredHeight hipDeviceAttributeMaxSurface2DLayered
#endif
#ifndef cudaDevAttrMaxSurface2DLayeredLayers
#  define cudaDevAttrMaxSurface2DLayeredLayers hipDeviceAttributeMaxSurface2DLayered
#endif
#ifndef cudaDevAttrMaxSurfaceCubemapWidth
#  define cudaDevAttrMaxSurfaceCubemapWidth hipDeviceAttributeMaxSurfaceCubemap
#endif
#ifndef cudaDevAttrMaxSurfaceCubemapLayeredWidth
#  define cudaDevAttrMaxSurfaceCubemapLayeredWidth hipDeviceAttributeMaxSurfaceCubemapLayered
#endif
#ifndef cudaDevAttrMaxSurfaceCubemapLayeredLayers
#  define cudaDevAttrMaxSurfaceCubemapLayeredLayers hipDeviceAttributeMaxSurfaceCubemapLayered
#endif
#ifndef cudaDevAttrMaxRegistersPerBlock
#  define cudaDevAttrMaxRegistersPerBlock hipDeviceAttributeMaxRegistersPerBlock
#endif
#ifndef cudaDevAttrMaxRegistersPerMultiprocessor
#  define cudaDevAttrMaxRegistersPerMultiprocessor hipDeviceAttributeMaxRegistersPerMultiprocessor
#endif
#ifndef cudaDevAttrTextureAlignment
#  define cudaDevAttrTextureAlignment hipDeviceAttributeTextureAlignment
#endif
#ifndef cudaDevAttrTexturePitchAlignment
#  define cudaDevAttrTexturePitchAlignment hipDeviceAttributeTexturePitchAlignment
#endif
#ifndef cudaDevAttrMultiProcessorCount
#  define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#endif
#ifndef cudaDevAttrPciBusId
#  define cudaDevAttrPciBusId hipDeviceAttributePciBusId
#endif
#ifndef cudaDevAttrPciDeviceId
#  define cudaDevAttrPciDeviceId hipDeviceAttributePciDeviceId
#endif
#ifndef cudaDevAttrComputeCapabilityMajor
#  define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#endif
#ifndef cudaDevAttrComputeCapabilityMinor
#  define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#endif
#ifndef cudaDevAttrMaxThreadsPerMultiProcessor
#  define cudaDevAttrMaxThreadsPerMultiProcessor hipDeviceAttributeMaxThreadsPerMultiProcessor
#endif
#ifndef cudaDevAttrMultiGpuBoardGroupID
#  define cudaDevAttrMultiGpuBoardGroupID hipDeviceAttributeMultiGpuBoardGroupID
#endif
#ifndef cudaDevAttrSingleToDoublePrecisionPerfRatio
#  define cudaDevAttrSingleToDoublePrecisionPerfRatio hipDeviceAttributeSingleToDoublePrecisionPerfRatio
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerBlockOptin
#  define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerMultiprocessor
#  define cudaDevAttrMaxSharedMemoryPerMultiprocessor hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#endif
#ifndef cudaDevAttrMaxAccessPolicyWindowSize
#  define cudaDevAttrMaxAccessPolicyWindowSize hipDeviceAttributeAccessPolicyMaxWindowSize
#endif
#ifndef cudaDevAttrMaxBlocksPerMultiprocessor
#  define cudaDevAttrMaxBlocksPerMultiprocessor hipDeviceAttributeMaxBlocksPerMultiProcessor
#endif
#ifndef cudaDevAttrMaxPersistingL2CacheSize
#  define cudaDevAttrMaxPersistingL2CacheSize hipDeviceAttributePersistingL2CacheMaxSize
#endif
#ifndef cudaDevAttrReservedSharedMemoryPerBlock
#  define cudaDevAttrReservedSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#endif
#ifndef cudaErrorInvalidValue
#  define cudaErrorInvalidValue hipErrorInvalidValue
#endif
#ifndef cudaErrorMemoryAllocation
#  define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#endif
#ifndef cudaSuccess
#  define cudaSuccess hipSuccess
#endif
#ifndef cudaErrorNotReady
#  define cudaErrorNotReady hipErrorNotReady
#endif
#ifndef cudaMemAllocationTypePinned
#  define cudaMemAllocationTypePinned hipMemAllocationTypePinned
#endif
#ifndef cudaMemHandleTypeNone
#  define cudaMemHandleTypeNone hipMemHandleTypeNone
#endif
#ifndef cudaMemHandleTypePosixFileDescriptor
#  define cudaMemHandleTypePosixFileDescriptor hipMemHandleTypePosixFileDescriptor
#endif
#ifndef cudaMemHandleTypeWin32
#  define cudaMemHandleTypeWin32 hipMemHandleTypeWin32
#endif
#ifndef cudaMemHandleTypeWin32Kmt
#  define cudaMemHandleTypeWin32Kmt hipMemHandleTypeWin32Kmt
#endif
#ifndef cudaMemLocation
#  define cudaMemLocation hipMemLocation
#endif
#ifndef cudaMemLocationTypeDevice
#  define cudaMemLocationTypeDevice hipMemLocationTypeDevice
#endif
#ifndef cudaMemLocationTypeHost
#  define cudaMemLocationTypeHost hipMemLocationTypeHost
#endif
#ifndef cudaMemPoolReuseAllowOpportunistic
#  define cudaMemPoolReuseAllowOpportunistic hipMemPoolReuseAllowOpportunistic
#endif
#ifndef cudaEventDefault
#  define cudaEventDefault hipEventDefault
#endif
#ifndef cudaEventDisableTiming
#  define cudaEventDisableTiming hipEventDisableTiming
#endif
#ifndef cudaEventBlockingSync
#  define cudaEventBlockingSync hipEventBlockingSync
#endif
#ifndef cudaEventInterprocess
#  define cudaEventInterprocess hipEventInterprocess
#endif
#ifndef cudaMemoryTypeDevice
#  define cudaMemoryTypeDevice hipMemoryTypeDevice
#endif
#ifndef cudaMemoryTypeHost
#  define cudaMemoryTypeHost hipMemoryTypeHost
#endif
#ifndef cudaMemoryTypeManaged
#  define cudaMemoryTypeManaged hipMemoryTypeManaged
#endif
#ifndef cudaMemoryTypeUnregistered
#  define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered
#endif
// functions
#ifndef cudaDeviceGetAttribute
#  define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif
#ifndef cudaDeviceGetDefaultMemPool
#  define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#endif
#ifndef cudaDeviceSynchronize
#  define cudaDeviceSynchronize hipDeviceSynchronize
#endif

#ifndef cudaDriverGetVersion
#  define cudaDriverGetVersion hipDriverGetVersion
#endif

#ifndef cudaEventCreateWithFlags
#  define cudaEventCreateWithFlags hipEventCreateWithFlags
#endif
#ifndef cudaEventDestroy
#  define cudaEventDestroy hipEventDestroy
#endif
#ifndef cudaEventRecord
#  define cudaEventRecord hipEventRecord
#endif
#ifndef cudaEventSynchronize
#  define cudaEventSynchronize hipEventSynchronize
#endif

#ifndef cudaFree
#  define cudaFree hipFree
#endif
#ifndef cudaFreeAsync
#  define cudaFreeAsync hipFreeAsync
#endif
#ifndef cudaFreeHost
#  define cudaFreeHost hipHostFree
#endif

#ifndef cudaGetDevice
#  define cudaGetDevice hipGetDevice
#endif
#ifndef cudaGetDeviceCount
#  define cudaGetDeviceCount hipGetDeviceCount
#endif
#ifndef cudaGetDeviceProperties
#define cudaGetDeviceProperties hipGetDeviceProperties
#endif
#ifndef cudaGetErrorName
#  define cudaGetErrorName hipGetErrorName
#endif
#ifndef cudaGetErrorString
#  define cudaGetErrorString hipGetErrorString
#endif
#ifndef cudaGetLastError
#  define cudaGetLastError hipGetLastError
#endif

#ifndef cudaMallocAsync
#  define cudaMallocAsync hipMallocAsync
#endif
#ifndef cudaMalloc
#  define cudaMalloc hipMalloc
#endif
#ifndef cudaMallocFromPoolAsync
#  define cudaMallocFromPoolAsync hipMallocFromPoolAsync
#endif
#ifndef cudaMallocHost
#  define cudaMallocHost hipHostMalloc
#endif
#ifndef cudaMallocManaged
#  define cudaMallocManaged hipMallocManaged
#endif

#ifndef cudaMemGetInfo
#  define cudaMemGetInfo hipMemGetInfo
#endif
#ifndef cudaMemPoolCreate
#  define cudaMemPoolCreate hipMemPoolCreate
#endif
#ifndef cudaMemPoolDestroy
#  define cudaMemPoolDestroy hipMemPoolDestroy
#endif
#ifndef cudaMemPoolSetAttribute
#  define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#endif

#ifndef cudaMemcpyAsync
#  define cudaMemcpyAsync hipMemcpyAsync
#endif
#ifndef cudaMemsetAsync
#  define cudaMemsetAsync hipMemsetAsync
#endif
#ifndef cudaMemset
#  define cudaMemset hipMemset
#endif

#ifndef cudaSetDevice
#  define cudaSetDevice hipSetDevice
#endif

#ifndef cudaStreamCreate
#  define cudaStreamCreate hipStreamCreate
#endif
#ifndef cudaStreamDestroy
#  define cudaStreamDestroy hipStreamDestroy
#endif
#ifndef cudaStreamSynchronize
#  define cudaStreamSynchronize hipStreamSynchronize
#endif

#ifndef cudaStreamWaitEvent
#  define cudaStreamWaitEvent(a,b,c) hipStreamWaitEvent(a,b,c)
#endif
#ifndef cudaEventCreate
#  define cudaEventCreate hipEventCreate
#endif
#ifndef cudaPointerGetAttributes
#  define cudaPointerGetAttributes hipPointerGetAttributes
#endif
#ifndef cudaEventElapsedTime
#  define cudaEventElapsedTime hipEventElapsedTime
#endif

#ifndef cudaStreamQuery
#  define cudaStreamQuery hipStreamQuery
#endif

#ifndef cudaStreamGetPriority
#  define cudaStreamGetPriority hipStreamGetPriority
#endif

#ifndef cudaHostAllocDefault
#  define cudaHostAllocDefault hipHostAllocDefault
#endif
#ifndef cudaHostAllocPortable
#  define cudaHostAllocPortable hipHostAllocPortable
#endif
#ifndef cudaHostAllocMapped
#  define cudaHostAllocMapped hipHostAllocMapped
#endif
#ifndef cudaHostAllocWriteCombined
#  define cudaHostAllocWriteCombined hipHostAllocWriteCombined
#endif
#ifndef cudaMemAttachGlobal
#  define cudaMemAttachGlobal hipMemAttachGlobal
#endif
#ifndef cudaMemAttachHost
#  define cudaMemAttachHost hipMemAttachHost
#endif

#ifndef cudaFlushGPUDirectRDMAWritesOptions
#  define cudaFlushGPUDirectRDMAWritesOptions hipFlushGPUDirectRDMAWritesOptions
#endif
#ifndef cudaFlushGPUDirectRDMAWritesOptionHost
#  define cudaFlushGPUDirectRDMAWritesOptionHost hipFlushGPUDirectRDMAWritesOptionHost
#endif
#ifndef cudaFlushGPUDirectRDMAWritesOptionMemOps
#  define cudaFlushGPUDirectRDMAWritesOptionMemOps hipFlushGPUDirectRDMAWritesOptionMemOps
#endif

#ifndef cudaGPUDirectRDMAWritesOrdering
#  define cudaGPUDirectRDMAWritesOrdering hipGPUDirectRDMAWritesOrdering
#endif
#ifndef cudaGPUDirectRDMAWritesOrderingNone
#  define cudaGPUDirectRDMAWritesOrderingNone hipGPUDirectRDMAWritesOrderingNone
#endif
#ifndef cudaGPUDirectRDMAWritesOrderingOwner
#  define cudaGPUDirectRDMAWritesOrderingOwner hipGPUDirectRDMAWritesOrderingOwner
#endif
#ifndef cudaGPUDirectRDMAWritesOrderingAllDevices
#  define cudaGPUDirectRDMAWritesOrderingAllDevices hipGPUDirectRDMAWritesOrderingAllDevices
#endif

#ifndef cudaDeviceCanAccessPeer
#  define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#endif

#ifndef HIPRT_CB
#  define HIPRT_CB
#endif
// NOTE(HIP/AMD): aliases for the CUDA driver-API types and enumerators
// used directly by upstream consumer code (physical_device.h,
// stream_ref.h, ensure_current_context.h, host_device_accessor.h,
// attributes.h, is_pointer_accessible.h, ...). The corresponding cuXxx
// function aliases are NOT provided here -- the libhipcxx driver-API
// wrappers in <amd/driver_api.h> call the HIP runtime / driver API
// functions directly using their native hipXxx names. Ported from
// upgrade/3.1.4.
#ifndef CUcontext
#  define CUcontext hipCtx_t
#endif
#ifndef CUdevice
#  define CUdevice hipDevice_t
#endif
#ifndef CUdevice_attribute
#  define CUdevice_attribute hipDeviceAttribute_t
#endif
#ifndef CUmemorytype
#  define CUmemorytype hipMemoryType
#endif
#ifndef CU_MEMORYTYPE_HOST
#  define CU_MEMORYTYPE_HOST hipMemoryTypeHost
#endif
#ifndef CU_MEMORYTYPE_DEVICE
#  define CU_MEMORYTYPE_DEVICE hipMemoryTypeDevice
#endif
#ifndef CU_POINTER_ATTRIBUTE_MEMORY_TYPE
#  define CU_POINTER_ATTRIBUTE_MEMORY_TYPE HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
#endif
#ifndef CU_POINTER_ATTRIBUTE_IS_MANAGED
#  define CU_POINTER_ATTRIBUTE_IS_MANAGED HIP_POINTER_ATTRIBUTE_IS_MANAGED
#endif

#ifndef CUstream
#  define CUstream hipStream_t
#endif
#ifndef CUresult
#  define CUresult hipError_t
#endif
#ifndef CUDA_SUCCESS
#  define CUDA_SUCCESS hipSuccess
#endif
#ifndef CUDA_CB
#  define CUDA_CB HIPRT_CB
#endif

#ifndef CUmemPool_attribute
#  define CUmemPool_attribute hipMemPoolAttr
#endif
#ifndef CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH
#  define CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH hipMemPoolAttrReservedMemHigh
#endif
#ifndef CU_MEMPOOL_ATTR_USED_MEM_HIGH
#  define CU_MEMPOOL_ATTR_USED_MEM_HIGH hipMemPoolAttrUsedMemHigh
#endif
#ifndef cudaMemPoolAttrReservedMemCurrent
#  define cudaMemPoolAttrReservedMemCurrent hipMemPoolAttrReservedMemCurrent
#endif
#ifndef cudaMemPoolAttrReservedMemHigh
#  define cudaMemPoolAttrReservedMemHigh hipMemPoolAttrReservedMemHigh
#endif
#ifndef cudaMemPoolAttrUsedMemCurrent
#  define cudaMemPoolAttrUsedMemCurrent hipMemPoolAttrUsedMemCurrent
#endif
#ifndef cudaMemPoolAttrUsedMemHigh
#  define cudaMemPoolAttrUsedMemHigh hipMemPoolAttrUsedMemHigh
#endif
#ifndef cudaMemPoolReuseAllowInternalDependencies
#  define cudaMemPoolReuseAllowInternalDependencies hipMemPoolReuseAllowInternalDependencies
#endif
#ifndef cudaMemPoolReuseFollowEventDependencies
#  define cudaMemPoolReuseFollowEventDependencies hipMemPoolReuseFollowEventDependencies
#endif
#ifndef cudaStreamNonBlocking
#  define cudaStreamNonBlocking hipStreamNonBlocking
#endif
#ifndef CUmemoryPool
#  define CUmemoryPool hipMemPool_t
#endif
#ifndef CUmemLocation
#  define CUmemLocation hipMemLocation
#endif
#ifndef CU_MEM_LOCATION_TYPE_DEVICE
#  define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#endif
#ifndef CU_MEM_LOCATION_TYPE_HOST
#  define CU_MEM_LOCATION_TYPE_HOST hipMemLocationTypeHost
#endif
#ifndef CUmemAccess_flags
#  define CUmemAccess_flags hipMemAccessFlags
#endif
#ifndef CU_MEM_ACCESS_FLAGS_PROT_READ
#  define CU_MEM_ACCESS_FLAGS_PROT_READ hipMemAccessFlagsProtRead
#endif
#ifndef CU_MEM_ACCESS_FLAGS_PROT_READWRITE
#  define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#endif
#ifndef cudaErrorNotSupported
#  define cudaErrorNotSupported hipErrorNotSupported
#endif
#ifndef CUdeviceptr
#  define CUdeviceptr hipDeviceptr_t
#endif
#ifndef CUmemAllocationHandleType
#  define CUmemAllocationHandleType hipMemAllocationHandleType
#endif
#ifndef CUmemAllocationType
#  define CUmemAllocationType hipMemAllocationType
#endif
// NOTE(HIP/AMD): managed memory pools are not supported by HIP. Map
// CU_MEM_ALLOCATION_TYPE_MANAGED to hipMemAllocationTypeMax (sentinel)
// so the corresponding upstream conditional checks (e.g. in
// memory_resource_base.h __get_pool_properties) compile but never
// match an actually-supported allocation type at runtime.
#ifndef CU_MEM_ALLOCATION_TYPE_MANAGED
#  define CU_MEM_ALLOCATION_TYPE_MANAGED hipMemAllocationTypeMax
#endif
#ifndef CU_MEM_LOCATION_TYPE_HOST_NUMA
#  define CU_MEM_LOCATION_TYPE_HOST_NUMA hipMemLocationTypeHostNuma
#endif
#ifndef CUmemAccessDesc
#  define CUmemAccessDesc hipMemAccessDesc
#endif
#ifndef CUmemPoolProps
#  define CUmemPoolProps hipMemPoolProps
#endif
#ifndef CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
#  define CU_MEMPOOL_ATTR_RELEASE_THRESHOLD hipMemPoolAttrReleaseThreshold
#endif
#ifndef CU_MEM_ALLOCATION_TYPE_PINNED
#  define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#endif
#ifndef CU_MEM_ACCESS_FLAGS_PROT_NONE
#  define CU_MEM_ACCESS_FLAGS_PROT_NONE hipMemAccessFlagsProtNone
#endif
#ifndef CUstreamCallback
#  define CUstreamCallback hipStreamCallback_t
#endif
#ifndef CUhostFn
#  define CUhostFn hipHostFn_t
#endif

#ifndef CUDART_CB
#  define CUDART_CB HIPRT_CB
#endif
#ifndef cudaStreamAddCallback
#  define cudaStreamAddCallback hipStreamAddCallback
#endif

#ifndef __nv_bfloat16
#  define __nv_bfloat16 __hip_bfloat16
#endif
#ifndef __nv_bfloat16_raw
#  define __nv_bfloat16_raw __hip_bfloat16_raw
#endif
#ifndef __nv_bfloat162
#  define __nv_bfloat162 __hip_bfloat162
#endif

// NOTE(HIP/AMD): map upstream __nv_fp{8,6,4}_* names to HIP's __hip_fp{8,6,4}_*
// equivalents. _CCCL_HAS_NVFP{8,6,4}() in <cuda/std/__cccl/extended_data_types.h>
// is enabled on HIP via these aliases; the HIP umbrella headers
// <hip/hip_fp{8,6,4}.h> are pulled in by
// <cuda/std/__floating_point/cuda_fp_types.h>.
#ifndef __nv_fp8_e4m3
#  define __nv_fp8_e4m3 __hip_fp8_e4m3
#endif
#ifndef __nv_fp8x2_e4m3
#  define __nv_fp8x2_e4m3 __hip_fp8x2_e4m3
#endif
#ifndef __nv_fp8x4_e4m3
#  define __nv_fp8x4_e4m3 __hip_fp8x4_e4m3
#endif
#ifndef __nv_fp8_e5m2
#  define __nv_fp8_e5m2 __hip_fp8_e5m2
#endif
#ifndef __nv_fp8x2_e5m2
#  define __nv_fp8x2_e5m2 __hip_fp8x2_e5m2
#endif
#ifndef __nv_fp8x4_e5m2
#  define __nv_fp8x4_e5m2 __hip_fp8x4_e5m2
#endif

#ifndef __nv_fp6_e2m3
#  define __nv_fp6_e2m3 __hip_fp6_e2m3
#endif
#ifndef __nv_fp6x2_e2m3
#  define __nv_fp6x2_e2m3 __hip_fp6x2_e2m3
#endif
#ifndef __nv_fp6x4_e2m3
#  define __nv_fp6x4_e2m3 __hip_fp6x4_e2m3
#endif
#ifndef __nv_fp6_e3m2
#  define __nv_fp6_e3m2 __hip_fp6_e3m2
#endif
#ifndef __nv_fp6x2_e3m2
#  define __nv_fp6x2_e3m2 __hip_fp6x2_e3m2
#endif
#ifndef __nv_fp6x4_e3m2
#  define __nv_fp6x4_e3m2 __hip_fp6x4_e3m2
#endif

#ifndef __nv_fp4_e2m1
#  define __nv_fp4_e2m1 __hip_fp4_e2m1
#endif
#ifndef __nv_fp4x2_e2m1
#  define __nv_fp4x2_e2m1 __hip_fp4x2_e2m1
#endif
#ifndef __nv_fp4x4_e2m1
#  define __nv_fp4x4_e2m1 __hip_fp4x4_e2m1
#endif

// NOTE(HIP/AMD): map upstream __nv_cvt_* fp8 conversion helpers and the
// __NV_E4M3 / __NV_E5M2 / __NV_NOSAT enum tags to HIP's __hip_cvt_* and
// __HIP_E4M3 / __HIP_E5M2 / __HIP_NOSAT equivalents. fp6 / fp4 / e8m0
// conversion helpers are not aliased because the corresponding fp6 / fp4
// type families are intentionally left disabled on HIP (see
// <cuda/std/__cccl/extended_data_types.h>) and e8m0 has no HIP analogue.
#ifndef __nv_saturation_t
#  define __nv_saturation_t __hip_saturation_t
#endif
#ifndef __nv_fp8_interpretation_t
#  define __nv_fp8_interpretation_t __hip_fp8_interpretation_t
#endif

#ifndef __NV_E4M3
#  define __NV_E4M3 __HIP_E4M3
#endif
#ifndef __NV_E5M2
#  define __NV_E5M2 __HIP_E5M2
#endif
#ifndef __NV_NOSAT
#  define __NV_NOSAT __HIP_NOSAT
#endif
#ifndef __NV_SATFINITE
#  define __NV_SATFINITE __HIP_SATFINITE
#endif

#ifndef __nv_cvt_float_to_fp8
#  define __nv_cvt_float_to_fp8 __hip_cvt_float_to_fp8
#endif
#ifndef __nv_cvt_double_to_fp8
#  define __nv_cvt_double_to_fp8 __hip_cvt_double_to_fp8
#endif
#ifndef __nv_cvt_halfraw_to_fp8
#  define __nv_cvt_halfraw_to_fp8 __hip_cvt_halfraw_to_fp8
#endif
#ifndef __nv_cvt_bfloat16raw_to_fp8
#  define __nv_cvt_bfloat16raw_to_fp8 __hip_cvt_bfloat16raw_to_fp8
#endif
#ifndef __nv_cvt_fp8_to_halfraw
#  define __nv_cvt_fp8_to_halfraw __hip_cvt_fp8_to_halfraw
#endif

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
// NOTE(HIP/AMD): pull <hip/hip_fp8.h> in here (same pattern as bf16/fp16
// above) so that '__hip_fp8_e4m3' / '__hip_fp8_e5m2' are complete types
// by the time '__nv_fp8_e4m3' / '__nv_fp8_e5m2' (defined as macros
// expanding to '__hip_fp8_*' below) reach a use site like
// 'is_same_v<_RawTp, __nv_fp8_e4m3>' in <cuda/std/__type_traits/num_bits.h>.
// Without this, TUs whose first include is '<cuda/__cccl_config>' (which
// pulls extended_data_types.h before <amd/cuda_runtime.h>) end up with
// '::__nv_fp8_e4m3' forward-declared instead of '::__hip_fp8_e4m3', and
// later macro expansion finds an undeclared name. fp4 / fp6 are
// intentionally NOT pulled in here -- their <hip/hip_fp{4,6}.h> headers
// have an internal-helper symbol clash, see
// /home/moberste/Coding/Reproducer/claude/hip_fp4_fp6_internal_helpers_redefinition.cpp.
#include <hip/hip_fp8.h>
__host__ __device__ __half __double2half(const double& __value) noexcept
{
  return __float2half(static_cast<float>(__value));
}

