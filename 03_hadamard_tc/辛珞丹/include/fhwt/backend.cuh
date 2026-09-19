// =============================================================================
//  backend.cuh -- the one place that picks the platform headers.
//
//  CUDA and MUSA are source compatible in the direction that matters here --
//  the same programming model, the same kernels -- but a MUSA installation
//  contains *no* cuda* identifiers at all: not the headers (musa_runtime.h
//  instead of cuda_runtime.h), not the types (musaStream_t instead of
//  cudaStream_t) and not the functions.  The vendor's migration path is a source
//  rewriting pass (`musify-text`), which is fine for a one-off port but leaves
//  the tree only compilable after a tool run.
//
//  This header takes the other route: everything in the code base stays spelled
//  with the CUDA names, and on MUSA the names are mapped to their musa*
//  equivalents here.  One checkout, two toolchains, no preprocessing step --
//  `mcc` compiles the same .cu files as `nvcc`.  (The handful of places where
//  the two runtimes genuinely differ in *behaviour* rather than in spelling are
//  handled where they occur, each with a comment: the Tensor Core stub in
//  kernel_tc.cuh, the five-argument musaGraphInstantiate, the missing
//  musaGraphUpload, the missing graph-edge query, and the packed bf16
//  intrinsics in types.cuh.)
//
//  Every mapping below was checked against MUSA 3.1's headers before being
//  written down; none of them is a guess.
// =============================================================================
#pragma once

#if defined(__MUSACC__)

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>

// The plugin types are the one place where MUSA uses a *different name for a
// different type* rather than a prefixed one: MUSA calls the bfloat16 scalar
// __mt_bfloat16, CUDA calls it __nv_bfloat16.  (__half / __half2 keep their
// names on both.)  Aliases keep the rest of the tree spelled the CUDA way.
using __nv_bfloat16 = __mt_bfloat16;
using __nv_bfloat162 = __mt_bfloat162;

#define cudaDeviceProp                              musaDeviceProp
#define cudaDeviceSynchronize                       musaDeviceSynchronize
#define cudaError_t                                 musaError_t
#define cudaEventCreate                             musaEventCreate
#define cudaEventCreateWithFlags                    musaEventCreateWithFlags
#define cudaEventDestroy                            musaEventDestroy
#define cudaEventDisableTiming                      musaEventDisableTiming
#define cudaEventElapsedTime                        musaEventElapsedTime
#define cudaEventRecord                             musaEventRecord
#define cudaEventSynchronize                        musaEventSynchronize
#define cudaEvent_t                                 musaEvent_t
#define cudaFree                                    musaFree
#define cudaFuncAttributeMaxDynamicSharedMemorySize musaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncSetAttribute                        musaFuncSetAttribute
#define cudaGetDeviceProperties                     musaGetDeviceProperties
#define cudaGetErrorString                          musaGetErrorString
#define cudaGetLastError                            musaGetLastError
#define cudaGraphAddChildGraphNode                  musaGraphAddChildGraphNode
#define cudaGraphAddKernelNode                      musaGraphAddKernelNode
#define cudaGraphCreate                             musaGraphCreate
#define cudaGraphDestroy                            musaGraphDestroy
#define cudaGraphExecDestroy                        musaGraphExecDestroy
#define cudaGraphExec_t                             musaGraphExec_t
#define cudaGraphGetEdges                           musaGraphGetEdges
#define cudaGraphGetNodes                           musaGraphGetNodes
#define cudaGraphInstantiate                        musaGraphInstantiate
#define cudaGraphKernelNodeGetParams                musaGraphKernelNodeGetParams
#define cudaGraphLaunch                             musaGraphLaunch
#define cudaGraphNodeGetType                        musaGraphNodeGetType
#define cudaGraphNodeType                           musaGraphNodeType
#define cudaGraphNodeTypeEmpty                      musaGraphNodeTypeEmpty
#define cudaGraphNodeTypeKernel                     musaGraphNodeTypeKernel
#define cudaGraphNode_t                             musaGraphNode_t
#define cudaGraphUpload                             musaGraphUpload
#define cudaGraph_t                                 musaGraph_t
#define cudaKernelNodeParams                        musaKernelNodeParams
#define cudaLaunchKernel                            musaLaunchKernel
#define cudaMalloc                                  musaMalloc
#define cudaMemcpy                                  musaMemcpy
#define cudaMemcpyAsync                             musaMemcpyAsync
#define cudaMemcpyDeviceToDevice                    musaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost                      musaMemcpyDeviceToHost
#define cudaMemcpyHostToDevice                      musaMemcpyHostToDevice
#define cudaMemset                                  musaMemset
#define cudaMemsetAsync                             musaMemsetAsync
#define cudaStreamBeginCapture                      musaStreamBeginCapture
#define cudaStreamCaptureModeThreadLocal            musaStreamCaptureModeThreadLocal
#define cudaStreamCreateWithFlags                   musaStreamCreateWithFlags
#define cudaStreamDestroy                           musaStreamDestroy
#define cudaStreamEndCapture                        musaStreamEndCapture
#define cudaStreamNonBlocking                       musaStreamNonBlocking
#define cudaStreamSynchronize                       musaStreamSynchronize
#define cudaStreamWaitEvent                         musaStreamWaitEvent
#define cudaStream_t                                musaStream_t
#define cudaSuccess                                 musaSuccess

#else

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#endif
