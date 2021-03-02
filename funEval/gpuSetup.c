#include <stdio.h>
#include <sys/stat.h>
#include "gpuSetup.h"

void gpuInit(struct gpuSetup *gpu, int dev)
{
    cl_int ret;

    ret = clGetPlatformIDs(1, &(gpu->plID), NULL);
    if(ret != CL_SUCCESS)
    {
        printf("Could not get platform.\n");
        exit(0);
    }

    cl_uint nDev;
    ret = clGetDeviceIDs(gpu->plID, CL_DEVICE_TYPE_ALL, 0, NULL, &(nDev));
    if(ret != CL_SUCCESS)
    {
        printf("Could not get number of devices.\n");
        exit(0);
    }
    cl_device_id dIDs[nDev];
    ret = clGetDeviceIDs(gpu->plID, CL_DEVICE_TYPE_ALL, nDev, dIDs, NULL);
    if(ret != CL_SUCCESS)
    {
        printf("Could not get device ids.\n");
        exit(0);
    }

    if(dev >= nDev)
    {
        printf("Device %d out of range. (%d)\n", dev, nDev);
        exit(0);
    }
    gpu->devID = dIDs[dev];

    gpu->context = clCreateContext(NULL, 1, &(gpu->devID), NULL, NULL, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create context\n");
        exit(0);
    }
    gpu->queue = clCreateCommandQueue(gpu->context, gpu->devID, 0, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create queue\n");
        exit(0);
    }

    struct stat st;
    stat(kernFilename, &st);
    size_t kernSize = st.st_size;

    char *kernSrc = (char *)malloc((kernSize+1)*sizeof(char));
    kernSrc[kernSize] = '\0';

    FILE *f = fopen(kernFilename, "r");
    fread(kernSrc, sizeof(char), kernSize, f);
    fclose(f);

    gpu->program = clCreateProgramWithSource(gpu->context, 1,
                                            (const char **)&kernSrc,
                                            (const size_t *)&kernSize, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create program\n");
        exit(0);
    }

#ifdef USE_DOUBLE
    ret = clBuildProgram(gpu->program, 1, &(gpu->devID),
                            "-Werror -D USE_DOUBLE",
                            NULL, NULL);
#else
    ret = clBuildProgram(gpu->program, 1, &(gpu->devID),
                            "-Werror",
                            NULL, NULL);
#endif
    if(ret != CL_SUCCESS)
    {
        printf("Could not build program: ");
        if(ret == CL_INVALID_PROGRAM)
            printf("invalid program\n");
        else if(ret == CL_INVALID_VALUE)
            printf("invalid value\n");
        else if(ret == CL_INVALID_DEVICE)
            printf("invalid device\n");
        else if(ret == CL_INVALID_BINARY)
            printf("invalid binary\n");
        else if(ret == CL_INVALID_BUILD_OPTIONS)
            printf("invalid build options\n");
        else if(ret == CL_INVALID_OPERATION)
            printf("invalid operation\n");
        else if(ret == CL_COMPILER_NOT_AVAILABLE)
            printf("compiler not available\n");
        else if(ret == CL_BUILD_PROGRAM_FAILURE)
            printf("build program failure\n");
        else if(ret == CL_OUT_OF_HOST_MEMORY)
            printf("out of host memory\n");
        else
            printf("error\n");

        size_t logSize;
        clGetProgramBuildInfo(gpu->program, gpu->devID, CL_PROGRAM_BUILD_LOG,
                                0, NULL, &logSize);
        char *log = (char *)malloc(logSize);
        clGetProgramBuildInfo(gpu->program, gpu->devID, CL_PROGRAM_BUILD_LOG,
                                logSize, log, NULL);
        printf("%s\n", log);
        free(log);


        exit(0);
    }

#ifdef USE_DOUBLE
    gpu->kern_feval = clCreateKernel(gpu->program, "kern_feval", &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create Kernel\n");
        exit(0);
    }
#endif

    gpu->kern_feval_f = clCreateKernel(gpu->program, "kern_feval_f", &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create Kernel\n");
        exit(0);
    }
}

cl_int gpuFree(struct gpuSetup *gpu)
{
    cl_int ret;
    ret = clFlush(gpu->queue);
    ret = clFinish(gpu->queue);
#ifdef USE_DOUBLE
    ret = clReleaseKernel(gpu->kern_feval);
#endif
    ret = clReleaseKernel(gpu->kern_feval_f);
    ret = clReleaseProgram(gpu->program);
    ret = clReleaseCommandQueue(gpu->queue);
    ret = clReleaseContext(gpu->context);

    return ret;
}
