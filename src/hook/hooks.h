#ifndef _HOOKS_H
#define _HOOKS_H
#include "cuda_runtime_api.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <iterator>
#include <ratio>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <unistd.h>
#include <nvml.h>
#include <cuda.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "http_server/server.hpp"

// unit definition
#define KB 			(size_t)1024
#define MB 			KB * KB
#define SEC_IN_MS 	1000L
#define SEC_IN_US 	1000L * SEC_IN_MS
#define SEC_IN_NS 	1000L * SEC_IN_US
#define US_IN_NS 	1000L
#define MS_IN_NS 	US_IN_NS * 1000L
#define MS_IN_US 	1000L
#define MAXPROC 	102400

// options
#define CGROUP_DIR getenv("CGROUP_DIR")


// function macro for checking and debugging
#define CUDA_CHECK(ret) cuda_assert((ret), false, __FILE__, __LINE__);
#define CUDA_CHECK_DONT_ABORT(ret)	 										\
	cuda_assert((ret), true, __FILE__, __LINE__);

#define NVML_CHECK(ret) nvml_assert((ret), false, __FILE__, __LINE__);
#define NVML_CHECK_DONT_ABORT(ret)  										\
	nvml_assert((ret), true, __FILE__, __LINE__);

// prototypes used for static variable definition
static size_t get_memory_limit();

// static variables
static volatile size_t gpu_mem_limit = get_memory_limit();

static volatile int running = 1;

void launch_http_server()  {
  const std::string address = "0.0.0.0";
  const std::string port = "8080";
  std::cout << "server thread" << std::endl;
  // Initialise the server.
  http::server::server s(address, port, ".", running);

  // Run the server until stopped.
  s.run();
}

// directly return the gpu memory capacity as, for now, we do not limit the
// memory usage.
static size_t get_memory_limit() { return 24220 * MB; }

inline void cuda_assert(
	CUresult code, bool suppress, const char *file, int line)
{
	if (code != CUDA_SUCCESS) {
		const char **err_str_p = NULL;
		const char **err_name_p = NULL;
		if (!suppress) {
			cuGetErrorString(code, err_str_p);
			cuGetErrorName(code, err_name_p);
			fprintf(stderr,"%s:%d: %s: %s\n",
					file, line, *err_name_p, *err_str_p);
			exit(code);
		}
	}
}

inline void nvml_assert(
	nvmlReturn_t code, bool suppress, const char *file, int line)
{
	if (code != NVML_SUCCESS) {
		if (!suppress) {
			fprintf(stderr,"%s:%d: %s\n", file, line, nvmlErrorString(code));
			exit(code);
		}
	}
}

static void read_pids(std::set<pid_t> &pids)
{
	char* cgroup_procs = (char *)malloc(sizeof(char) * 100);
	snprintf(cgroup_procs, 100, "%s/cgroup.procs", CGROUP_DIR);
    std::ifstream fs(cgroup_procs);
    for(std::string line; std::getline(fs, line); )
        pids.insert(atoi(line.c_str()));
    fs.close();
	free(cgroup_procs);
}

static int get_gpu_compute_processes(
		uint32_t *proc_count, nvmlProcessInfo_t *proc_infos)
{
    nvmlReturn_t ret;
    nvmlDevice_t device;
    NVML_CHECK(
		nvmlDeviceGetHandleByIndex(0, &device));
    NVML_CHECK(
		nvmlDeviceGetComputeRunningProcesses(device, proc_count, proc_infos));
    return 0;
}


static CUresult validate_memory(size_t to_allocate)
{
    CUresult cu_res = CUDA_SUCCESS;
    size_t totalUsed = 0;

    // TODO handle race condition
    if (totalUsed + to_allocate > gpu_mem_limit)
		return CUDA_ERROR_OUT_OF_MEMORY;

	return cu_res;
}

static size_t get_size_of(CUarray_format fmt)
{
    size_t byte_sz = 1;
    switch (fmt) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
    case CU_AD_FORMAT_SIGNED_INT8:
    case CU_AD_FORMAT_NV12:
        byte_sz = 1;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
    case CU_AD_FORMAT_SIGNED_INT16:
    case CU_AD_FORMAT_HALF:
        byte_sz = 2;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT32:
    case CU_AD_FORMAT_SIGNED_INT32:
    case CU_AD_FORMAT_FLOAT:
        byte_sz = 4;
        break;
    }
    return byte_sz;
}
#endif  // _HOOKS_H
