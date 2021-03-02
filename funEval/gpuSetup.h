#ifndef FEVAL_GPUSETUP
#define FEVAL_GPUSETUP

#include <OpenCL/opencl.h>

const static char kernFilename[] = "funEvalKernel.cl";

struct gpuSetup
{
    cl_platform_id plID;
    cl_device_id devID;
    cl_program program;
    cl_context context;
    cl_command_queue queue;

    cl_kernel kern_feval;
    cl_kernel kern_feval_f;
};

void gpuInit(struct gpuSetup *gpu, int dev);
void gpuFree(struct gpuSetup *gpu);

#endif
