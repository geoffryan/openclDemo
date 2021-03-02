#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <OpenCL/opencl.h>
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
        f[i] = x[i]*x[i] - 3*x[i] + 1.0;
    clock_t end = clock();

    return end-start;
}

clock_t runOCL_f(float *x, float *f, int N, struct gpuSetup *gpu)
{
    cl_int ret;
    size_t size = N*sizeof(float);
    cl_mem x_mem_obj = clCreateBuffer(gpu->context, CL_MEM_READ_ONLY,
                                      size, NULL, &ret);
    cl_mem f_mem_obj = clCreateBuffer(gpu->context, CL_MEM_READ_WRITE,
                                      size, NULL, &ret);

    clock_t start = clock();

    ret = clEnqueueWriteBuffer(gpu->queue, x_mem_obj, CL_FALSE, 0, size,
                               x, 0, NULL, NULL);

    ret = clSetKernelArg(gpu->kern_feval_f, 0, sizeof(cl_mem),
                         (void *)&x_mem_obj);
    ret = clSetKernelArg(gpu->kern_feval_f, 1, sizeof(cl_mem),
                         (void *)&f_mem_obj);

    size_t workSize = N;
    size_t groupSize = 32;

    ret = clEnqueueNDRangeKernel(gpu->queue, gpu->kern_feval_f, 1, NULL,
                                 &workSize, &groupSize, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(gpu->queue, f_mem_obj, CL_FALSE, 0, size,
                              f, 0, NULL, NULL);

    clock_t end = clock();

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
    cl_mem f_mem_obj = clCreateBuffer(gpu->context, CL_MEM_READ_WRITE,
                                      size, NULL, &ret);

    clock_t start = clock();

    ret = clEnqueueWriteBuffer(gpu->queue, x_mem_obj, CL_FALSE, 0, size,
                               x, 0, NULL, NULL);

    ret = clSetKernelArg(gpu->kern_feval, 0, sizeof(cl_mem),
                         (void *)&x_mem_obj);
    ret = clSetKernelArg(gpu->kern_feval, 1, sizeof(cl_mem),
                         (void *)&f_mem_obj);

    size_t workSize = N;
    size_t groupSize = 32;

    ret = clEnqueueNDRangeKernel(gpu->queue, gpu->kern_feval, 1, NULL,
                                 &workSize, &groupSize, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(gpu->queue, f_mem_obj, CL_FALSE, 0, size,
                              f, 0, NULL, NULL);

    clock_t end = clock();

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

    int N = (int) strtol(argv[1], NULL, 10);

    struct gpuSetup gpu0;
    struct gpuSetup gpu1;
    struct gpuSetup gpu2;

    double *x = (double *)malloc(N * sizeof(double));
    double *fS = (double *)malloc(N * sizeof(double));
    double *f0 = (double *)malloc(N * sizeof(double));
    double *f1 = (double *)malloc(N * sizeof(double));
    double *f2 = (double *)malloc(N * sizeof(double));

    float *x_f = (float *)malloc(N * sizeof(float));
    float *fS_f = (float *)malloc(N * sizeof(float));
    float *f0_f = (float *)malloc(N * sizeof(float));
    float *f1_f = (float *)malloc(N * sizeof(float));
    float *f2_f = (float *)malloc(N * sizeof(float));

    int i;
    double xmax = 10.0;
    for(i=0; i<N; i++)
    {
        x[i] = (i*xmax)/(N-1);
        x_f[i] = (float)x[i];
    }

    gpuInit(&gpu0, 0);
    gpuInit(&gpu1, 1);
    gpuInit(&gpu2, 2);
  
    clock_t cS = runSerial(x, fS, N);
    clock_t c0 = runOCL(x, f0, N, &gpu0);
    clock_t c1 = runOCL(x, f1, N, &gpu1);
    clock_t c2 = runOCL(x, f2, N, &gpu2);
  
    clock_t cS_f = runSerial_f(x_f, fS_f, N);
    clock_t c0_f = runOCL_f(x_f, f0_f, N, &gpu0);
    clock_t c1_f = runOCL_f(x_f, f1_f, N, &gpu1);
    clock_t c2_f = runOCL_f(x_f, f2_f, N, &gpu2);

    double errS = L1(x, fS, fS, N);
    double err0 = L1(x, fS, f0, N);
    double err1 = L1(x, fS, f1, N);
    double err2 = L1(x, fS, f2, N);

    double errS_f = L1_f(x_f, fS_f, fS_f, N);
    double err0_f = L1_f(x_f, fS_f, f0_f, N);
    double err1_f = L1_f(x_f, fS_f, f1_f, N);
    double err2_f = L1_f(x_f, fS_f, f2_f, N);

    if(N <= 32)
    {
        printf("Serial:");
        for(i=0; i<N; i++)
            printf(" %.3lf", fS[i]);
        printf("\n");

        printf("Dev0:");
        for(i=0; i<N; i++)
            printf(" %.3lf", f0[i]);
        printf("\n");

        printf("Dev1:");
        for(i=0; i<N; i++)
            printf(" %.3lf", f1[i]);
        printf("\n");

        printf("Dev2:");
        for(i=0; i<N; i++)
            printf(" %.3lf", f2[i]);
        printf("\n\n");
        
        printf("Serial:");
        for(i=0; i<N; i++)
            printf(" %.3f", fS_f[i]);
        printf("\n");

        printf("Dev0:");
        for(i=0; i<N; i++)
            printf(" %.3f", f0_f[i]);
        printf("\n");

        printf("Dev1:");
        for(i=0; i<N; i++)
            printf(" %.3f", f1_f[i]);
        printf("\n");

        printf("Dev2:");
        for(i=0; i<N; i++)
            printf(" %.3f", f2_f[i]);
        printf("\n\n");
    }


    printf("Serial: %.1e s  (%.2e)\n", cS / ((double) CLOCKS_PER_SEC), errS);
    printf("Dev0:   %.1e s  (%.2e)\n", c0 / ((double) CLOCKS_PER_SEC), err0);
    printf("Dev1:   %.1e s  (%.2e)\n", c1 / ((double) CLOCKS_PER_SEC), err1);
    printf("Dev2:   %.1e s  (%.2e)\n", c2 / ((double) CLOCKS_PER_SEC), err2);

    printf("Serial: %.1e s  (%.2e)\n", cS_f / ((double) CLOCKS_PER_SEC), errS_f);
    printf("Dev0:   %.1e s  (%.2e)\n", c0_f / ((double) CLOCKS_PER_SEC), err0_f);
    printf("Dev1:   %.1e s  (%.2e)\n", c1_f / ((double) CLOCKS_PER_SEC), err1_f);
    printf("Dev2:   %.1e s  (%.2e)\n", c2_f / ((double) CLOCKS_PER_SEC), err2_f);
    
    gpuFree(&gpu0);
    gpuFree(&gpu1);
    gpuFree(&gpu2);

    free(x);
    free(fS);
    free(f0);
    free(f1);
    free(f2);

    return 0;
}
