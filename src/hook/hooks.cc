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

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <pthread.h>
#include "hooks.h"
#include "nvml.h"
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <nvtx3/nvToolsExt.h>

using namespace std;
#define _SYNC_BEFORE_PREEMPT
#ifdef _SCHEDULER_LOCK

static string suffix(getenv("SUFFIX")); // TODO: bug when genenv returns NULL
static string named_mtx_name("named_mutex_" + suffix);
static string named_cnd_name("named_cnd_" + suffix);
static string named_mtx_dev_sync_name("named_mutex_dev_sync_" + suffix);
static string named_cnd_dev_sync_name("named_cnd_dev_sync_" + suffix);

static std::shared_ptr<boost::interprocess::shared_memory_object> shm_ptr;
static std::shared_ptr<boost::interprocess::mapped_region> region_ptr;
static boost::interprocess::named_mutex named_mtx(
    boost::interprocess::open_only, named_mtx_name.c_str());
static boost::interprocess::named_condition named_cnd(
    boost::interprocess::open_only, named_cnd_name.c_str());

#ifdef _SYNC_BEFORE_PREEMPT
static boost::interprocess::named_mutex named_mtx_dev_sync(
    boost::interprocess::open_only, named_mtx_dev_sync_name.c_str());
static boost::interprocess::named_condition named_cnd_dev_sync(
    boost::interprocess::open_only, named_cnd_dev_sync_name.c_str());
static volatile int *gpu_empty;
#endif // _SYNC_BEFORE_PREEMPT

static volatile int *current_process;


void init_shared_mem() {
    string shm_name("MySharedMemory_" + suffix);
    shm_ptr = make_shared<boost::interprocess::shared_memory_object>(
        boost::interprocess::open_or_create, shm_name.c_str(),
        boost::interprocess::read_write);
    region_ptr = make_shared<boost::interprocess::mapped_region>(
        *shm_ptr, boost::interprocess::read_write);

    int *mem = static_cast<int*>(region_ptr->get_address());

    current_process = &mem[0];
#ifdef _SYNC_BEFORE_PREEMPT
    gpu_empty = &mem[1];
#endif // _SYNC_BEFORE_PREEMPT
}

#endif

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
    // printf("%d %d\n", kernel_launch_time, get_id());

#ifdef _GROUP_EVENT
    if (kernel_launch_time == 0) {
        for (int i = 0; i < EVENT_POOL_SIZE; i++) {
            cudaEventCreate(&cu_event_cycle[i]);
        }
    }
    static int counter = 0;
    if (kernel_launch_time % queue_group_size == 0) {
        if (kernel_launch_time / queue_group_size > 2) {
            int prev_idx = (cur_event_idx - 2 + EVENT_POOL_SIZE) % EVENT_POOL_SIZE;
            while((counter++) % 100 != 0 || cudaEventQuery(cu_event_cycle[prev_idx]) != cudaSuccess) {
                // printf("waitting\n");
                // wait
            }
        }
    }

    kernel_launch_time++;
    // printf("%d %d\n", kernel_launch_time, get_id());

#endif

#ifdef _SCHEDULER_LOCK

    if(shm_ptr == NULL || region_ptr == NULL) {
        init_shared_mem();
    }
    if (get_id() == 0) {
        //job 1 is running, give way.
        // https://www.boost.org/doc/libs/1_63_0/doc/html/thread/synchronization.html#thread.synchronization.condvar_ref
        bool pushed = false;
        boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(named_mtx);
        while(*current_process == 1) {
#ifdef _SYNC_BEFORE_PREEMPT
            cuStreamSynchronize(hStream);
            {
                boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock_dev_sync(named_mtx_dev_sync);
                *gpu_empty = 1;
                named_cnd_dev_sync.notify_one();
            }
#endif // _SYNC_BEFORE_PREEMPT
            if (!pushed) {
                nvtxRangePushA("preemption");
                pushed = true;
            }
            named_cnd.wait(lock);
        }
        if (pushed) {
            nvtxRangePop();
        }
#ifdef _SYNC_BEFORE_PREEMPT
        boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock_dev_sync(named_mtx_dev_sync);
        *gpu_empty = 0;
#endif // _SYNC_BEFORE_PREEMPT
    }
#endif

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

#ifdef _GROUP_EVENT
    if (kernel_launch_time % queue_group_size == 0) {
        cudaEventRecord(cu_event_cycle[cur_event_idx % EVENT_POOL_SIZE]);
        cur_event_idx++;
    }
#endif

#ifdef _SYNC_QUEUE
    //only add synchronization point to the long job.
    static int cnt = 0;
    if (cnt++ % SYNC_KERNELS == 0) {
        nvtxRangePushA("sync");
        cuStreamSynchronize(hStream);
        nvtxRangePop();
    }
#endif
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
