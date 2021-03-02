#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CL_TARGET_OPENCL_VERSION 110
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "gpuSetup.h"

clock_t runSerial(double *x, double *f, int N)
{
    clock_t start = clock();
    int i;
    for(i=0; i<N; i++)
        f[i] = x[i]*x[i] - 3*x[i] + 1.0;
    clock_t end = clock();

    return end-start;
}

clock_t runSerial_f(float *x, float *f, int N)
{
    clock_t start = clock();
    int i;
    for(i=0; i<N; i++)
        f[i] = x[i]*x[i] - 3.0f*x[i] + 1.0f;
    clock_t end = clock();

    return end-start;
}

clock_t runOCL_f(float *x, float *f, int N, struct gpuSetup *gpu)
{
    cl_int ret;
    size_t size = N*sizeof(float);
    cl_mem x_mem_obj = clCreateBuffer(gpu->context, CL_MEM_READ_ONLY,
                                      size, NULL, &ret);
    cl_mem f_mem_obj = clCreateBuffer(gpu->context, CL_MEM_WRITE_ONLY,
                                      size, NULL, &ret);


    ret = clEnqueueWriteBuffer(gpu->queue, x_mem_obj, CL_TRUE, 0, size,
                               (void *)x, 0, NULL, NULL);

    ret = clSetKernelArg(gpu->kern_feval_f, 0, sizeof(cl_mem),
                         (void *)&x_mem_obj);
    ret = clSetKernelArg(gpu->kern_feval_f, 1, sizeof(cl_mem),
                         (void *)&f_mem_obj);

    size_t workSize = N;
    size_t groupSize = 16;

    clock_t start = clock();
    ret = clEnqueueNDRangeKernel(gpu->queue, gpu->kern_feval_f, 1, NULL,
                                 &workSize, &groupSize,
                                 0, NULL, NULL);
    clock_t end = clock();
    ret = clEnqueueReadBuffer(gpu->queue, f_mem_obj, CL_TRUE, 0, size,
                              (void *)f, 0, NULL, NULL);


    clReleaseMemObject(x_mem_obj);
    clReleaseMemObject(f_mem_obj);

    return end-start;
}

clock_t runOCL(double *x, double *f, int N, struct gpuSetup *gpu)
{
    cl_int ret;
    size_t size = N*sizeof(double);
    cl_mem x_mem_obj = clCreateBuffer(gpu->context, CL_MEM_READ_ONLY,
                                      size, NULL, &ret);
    cl_mem f_mem_obj = clCreateBuffer(gpu->context, CL_MEM_WRITE_ONLY,
                                      size, NULL, &ret);


    ret = clEnqueueWriteBuffer(gpu->queue, x_mem_obj, CL_TRUE, 0, size,
                               x, 0, NULL, NULL);

    ret = clSetKernelArg(gpu->kern_feval, 0, sizeof(cl_mem),
                         (void *)&x_mem_obj);
    ret = clSetKernelArg(gpu->kern_feval, 1, sizeof(cl_mem),
                         (void *)&f_mem_obj);

    size_t workSize = N;
    size_t groupSize = 32;
    clock_t start = clock();

    ret = clEnqueueNDRangeKernel(gpu->queue, gpu->kern_feval, 1, NULL,
                                 &workSize, &groupSize, 0, NULL, NULL);
    clock_t end = clock();
    ret = clEnqueueReadBuffer(gpu->queue, f_mem_obj, CL_TRUE, 0, size,
                              f, 0, NULL, NULL);


    clReleaseMemObject(x_mem_obj);
    clReleaseMemObject(f_mem_obj);

    return end-start;
}

double L1(double *x, double *f1, double *f2, int N)
{
    double sum0 = 0.0;
    double sum = 0.0;
    int i;
    for(i=0; i<N-1; i++)
    {
        sum0 += 0.5*(fabs(f1[i])+fabs(f1[i+1])) * (x[i+1]-x[i]);
        sum += 0.5*(fabs(f1[i]-f2[i])+fabs(f1[i+1]-f2[i+1])) * (x[i+1]-x[i]);
    }

    return sum / sum0;
}

double L1_f(float *x, float *f1, float *f2, int N)
{
    double sum0 = 0.0;
    double sum = 0.0;
    int i;
    for(i=0; i<N-1; i++)
    {
        sum0 += 0.5*(fabs(f1[i])+fabs(f1[i+1])) * (x[i+1]-x[i]);
        sum += 0.5*(fabs(f1[i]-f2[i])+fabs(f1[i+1]-f2[i+1])) * (x[i+1]-x[i]);
    }

    return sum / sum0;
}

int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        printf("GIVE ME A NUMBER\n");
        return 0;
    }

    cl_uint numDev;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDev);
    cl_uint j;
        
    char fname_serial[] = "serial_results.txt";
    FILE *file = fopen(fname_serial, "w");
    fclose(file);

    for(j=0; j<numDev; j++)
    {
        char fname[256];
        sprintf(fname, "dev%llu_results.txt", (unsigned long long) j);
        file = fopen(fname, "w");
        fclose(file);
    }

    int Na = (int) strtol(argv[1], NULL, 10);
    int N = 32;
    while(2*N <= Na)
    {
        double *xd = (double *)malloc(N * sizeof(double));
        double *fSd = (double *)malloc(N * sizeof(double));
        double *fd = (double *)malloc(N * sizeof(double));

        float *xf = (float *)malloc(N * sizeof(float));
        float *fSf = (float *)malloc(N * sizeof(float));
        float *ff = (float *)malloc(N * sizeof(float));

        int i;
        double xmax = 10.0;
        for(i=0; i<N; i++)
        {
            xd[i] = (i*xmax)/(N-1);
            xf[i] = (float)xd[i];
        }
        
        clock_t cSf = runSerial_f(xf, fSf, N);
        clock_t cSd = runSerial(xd, fSd, N);
        if(N <= 32)
        {
            printf("Serial (single):");
            for(i=0; i<N; i++)
                printf(" %.3lf", fSf[i]);
            printf("\n");
            printf("Serial (double):");
            for(i=0; i<N; i++)
                printf(" %.3lf", fSd[i]);
            printf("\n");
        }
        printf("Serial (single): %.1e s\n", cSf / ((double) CLOCKS_PER_SEC));
        printf("Serial (double): %.1e s\n", cSd / ((double) CLOCKS_PER_SEC));

        file = fopen(fname_serial, "a");
        fprintf(file, "%d %.6e\n",
                N, cSf / ((double) CLOCKS_PER_SEC));
        fclose(file);

        for(j=0; j<numDev; j++)
        {
            printf("\nDevice %llu\n", (unsigned long long) j);
            for(i=0; i<N; i++)
            {
                fd[i] = 0.0;
                ff[i] = 0.0f;
            }
            struct gpuSetup gpu;
            gpuInit(&gpu, j);

            clock_t cf = runOCL_f(xf, ff, N, &gpu);
            double errf = L1_f(xf, fSf, ff, N);
            
#ifdef USE_DOUBLE
            clock_t cd = runOCL_f(xd, fd, N, &gpu);
            double errd = L1(xd, fSd, fd, N);
#endif
        
            if(N <= 32)
            {
                printf("Dev%llu (single):", (unsigned long long) j);
                for(i=0; i<N; i++)
                    printf(" %.3lf", ff[i]);
                printf("\n");
#ifdef USE_DOUBLE
                printf("Dev%llu (double):", (unsigned long long) j);
                for(i=0; i<N; i++)
                    printf(" %.3lf", fd[i]);
                printf("\n");
#endif
            }
            printf("Dev%llu (single):   %.1e s  (%.2e)\n",
                    (unsigned long long) j,
                   cf / ((double) CLOCKS_PER_SEC), errf);
#ifdef USE_DOUBLE
            printf("Dev%llu (double):   %.1e s  (%.2e)\n",
                    (unsigned long long) j,
                   cd / ((double) CLOCKS_PER_SEC), errd);
#endif
        
            gpuFree(&gpu);

            char fname[256];
            sprintf(fname, "dev%llu_results.txt", (unsigned long long) j);
            file = fopen(fname, "a");
            fprintf(file, "%d %.6e %.6e\n",
                    N, cf / ((double) CLOCKS_PER_SEC), errf);
            fclose(file);
        }

        free(xf);
        free(xd);
        free(fSf);
        free(fSd);
        free(ff);
        free(fd);

        N *= 2;
    }


    return 0;
}
