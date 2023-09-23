/*
Copyright (c) 2022 Futurewei Technologies.
Author: Hao Xu (@hxhp)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iomanip>
#include "hooks.h"
#include "nvml.h"

/*************************** hooks functions below ***************************/
CUresult cuInit_hook(uint32_t flags)
{
    CUresult cures = CUDA_SUCCESS;
    return cures;
}

CUresult cuInit_posthook(uint32_t flags)
{
    CUresult cures = CUDA_SUCCESS;
    return cures;
}

CUresult cuMemAlloc_hook(CUdeviceptr* dptr, size_t byte_sz)
{
    return validate_memory(byte_sz);
}

CUresult cuMemAllocManaged_hook(CUdeviceptr *dptr, size_t byte_sz, uint32_t
		flags)
{
    return validate_memory(byte_sz);
}

CUresult cuMemAllocPitch_hook(CUdeviceptr *dptr, size_t *pPitch, size_t
		WidthInBytes, size_t Height, uint32_t ElementSizeBytes)
{

    size_t toAllocate = WidthInBytes * Height / 100 * 101;
    return validate_memory(toAllocate);
}

CUresult cuArrayCreate_hook(
		CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
    size_t height = pAllocateArray->Height;
    size_t toAllocate;

    height = (height == 0) ? 1 : height;
    toAllocate = pAllocateArray->NumChannels * pAllocateArray->Width * height;
    toAllocate *= get_size_of(pAllocateArray->Format);
    return validate_memory(toAllocate);
}

CUresult cuArray3DCreate_hook(
		CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
{
    size_t depth = pAllocateArray->Depth;
    size_t height = pAllocateArray->Height;
    size_t toAllocate;

    height = (height == 0) ? 1 : height;
    depth = (depth == 0) ? 1 : depth;
    toAllocate =
		pAllocateArray->NumChannels * pAllocateArray->Width * height * depth;
    toAllocate *= get_size_of(pAllocateArray->Format);
    return validate_memory(toAllocate);
}

CUresult cuMipmappedArrayCreate_hook(
		CUmipmappedArray *pHandle,
		const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
		uint32_t numMipmapLevels)
{
    size_t depth = pMipmappedArrayDesc->Depth;
    size_t height = pMipmappedArrayDesc->Height;
    size_t toAllocate;

    height = (height == 0) ? 1 : height;
    depth = (depth == 0) ? 1 : depth;
    toAllocate = pMipmappedArrayDesc->NumChannels * pMipmappedArrayDesc->Width
		* height * depth;
    toAllocate *= get_size_of(pMipmappedArrayDesc->Format);
    return validate_memory(toAllocate);
}

/**
 * Wenqing: Logic executed before intercepted cuLaunchKernel CUDA call.
*/
CUresult cuLaunchKernel_hook(
		CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
		uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
		uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream,
		void** kernelParams, void** extra)
{
    CUresult cures = CUDA_SUCCESS;
	uint32_t cost = gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY *
		blockDimZ;
    // kernel_launch_time++;


    if(kernel_launch_time == 1) {
        cudaEventCreate(&cu_dummy);
    }

    return cures;
}

/**
 * Wenqing: Logic executed after cuLaunchKernel CUDA call.
*/
CUresult cuLaunchKernel_posthook(
		CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
		uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
		uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream,
		void** kernelParams, void** extra)
{
	CUresult ret = CUDA_SUCCESS;

	return ret;
}

CUresult cuLaunchCooperativeKernel_hook(
		CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
		uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
		uint32_t blockDimZ, uint32_t sharedMemBytes,
		CUstream hStream, void **kernelParams)
{
    return cuLaunchKernel_hook(
			f,
			gridDimX, gridDimY, gridDimZ,
			blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, hStream, kernelParams, NULL);
}

CUresult cuLaunchCooperativeKernel_posthook(
		CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
		uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
		uint32_t blockDimZ, uint32_t sharedMemBytes,
		CUstream hStream, void **kernelParams)
{
    return cuLaunchKernel_posthook(
			f,
			gridDimX, gridDimY, gridDimZ,
			blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, hStream, kernelParams, NULL);
}

CUresult cuDeviceTotalMem_posthook(size_t* bytes, CUdevice dev)
{
    *bytes = gpu_mem_limit;
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo_posthook(size_t* free, size_t* total)
{
    // get process ids within the same container
    std::set<pid_t> pids;
    read_pids(pids);

    // get per process gpu memory usage
    uint32_t procCount = MAXPROC;
    nvmlProcessInfo_t procInfos[MAXPROC];
    size_t totalUsed = 0;
    int ret = get_gpu_compute_processes(&procCount, procInfos);
    if (ret != 0) {
        return CUDA_SUCCESS;
    }

    for (int i = 0; i < procCount; ++i) {
        uint32_t pid = procInfos[i].pid;
        if (pids.find(pid) != pids.end())
			totalUsed += procInfos[i].usedGpuMemory;
    }

    *total = gpu_mem_limit;
    *free = totalUsed > gpu_mem_limit ? 0 : gpu_mem_limit - totalUsed;
    return CUDA_SUCCESS;
}
