#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 110
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


void printInfoCLUINT(cl_device_id dev, cl_device_info attr, char name[])
{
    cl_uint info;
    clGetDeviceInfo(dev, attr, sizeof(cl_uint), &info, NULL);
    printf("        %s: %u\n", name, info);
}

void printInfoCLULONG(cl_device_id dev, cl_device_info attr, char name[])
{
    cl_ulong info;
    clGetDeviceInfo(dev, attr, sizeof(cl_ulong), &info, NULL);
    printf("        %s: %llu\n", name, (unsigned long long)info);
}

void printInfoCLBOOL(cl_device_id dev, cl_device_info attr, char name[])
{
    cl_bool info;
    clGetDeviceInfo(dev, attr, sizeof(cl_bool), &info, NULL);
    if(info)
        printf("        %s: true\n", name);
    else
        printf("        %s: false\n", name);
}

void printInfoSIZET(cl_device_id dev, cl_device_info attr, char name[])
{
    size_t info;
    clGetDeviceInfo(dev, attr, sizeof(size_t), &info, NULL);
    printf("        %s: %zu\n", name, info);
}

void printInfoSTR(cl_device_id dev, cl_device_info attr, char name[])
{
    size_t size;
    cl_int ret = clGetDeviceInfo(dev, attr, 0, NULL, &size);
    char *info = (char *) malloc(size);

    clGetDeviceInfo(dev, attr, size, info, NULL);

    printf("        %s: %s\n", name, info);
    free(info);
}



int main(int argc, char *argv[])
{
    cl_uint nPlatforms;
    clGetPlatformIDs(0, NULL, &nPlatforms);
    cl_platform_id platform[nPlatforms];
    clGetPlatformIDs(nPlatforms, platform, NULL);

    const char *platformAttributeName[5] = {"Name", "Vendor", "Version",
                                                "Profile", "Extensions"};
    cl_platform_info platformAttribute[5] = {CL_PLATFORM_NAME,
                                              CL_PLATFORM_VENDOR,
                                              CL_PLATFORM_VERSION,
                                              CL_PLATFORM_PROFILE,
                                              CL_PLATFORM_EXTENSIONS};

    int i;
    for(i=0; i<nPlatforms; i++)
    {
        printf("Platform %d:\n", i);

        int j;
        for(j=0; j<5; j++)
        {
            size_t attrSize;
            clGetPlatformInfo(platform[i], platformAttribute[j], 0, NULL,
                                &attrSize);
            char *info = (char *)malloc(attrSize);
            clGetPlatformInfo(platform[i], platformAttribute[j], attrSize,
                                info, NULL);
            printf("    %s: %s\n", platformAttributeName[j], info);
            free(info);
        }

        cl_uint nDevices;
        clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &nDevices);
        cl_device_id device[nDevices];
        clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, nDevices, device,
                        NULL);

        for(j=0; j<nDevices; j++)
        {
            cl_uint info_uint;

            printf("    Device %d:\n", j);
            cl_device_fp_config info_fp_config;
            cl_device_exec_capabilities info_exec_capabilities;
            cl_device_mem_cache_type info_mem_cache_type;
            cl_device_local_mem_type info_local_mem_type;
            cl_command_queue_properties info_command_queue_props;
            cl_device_type info_type;

            printInfoCLUINT(device[j], CL_DEVICE_ADDRESS_BITS, "Address Bits");
            printInfoCLBOOL(device[j], CL_DEVICE_AVAILABLE, "Available");
            printInfoCLBOOL(device[j], CL_DEVICE_COMPILER_AVAILABLE,
                            "Compiler Available");

#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
            clGetDeviceInfo(device[j], CL_DEVICE_DOUBLE_FP_CONFIG,
                            sizeof(cl_device_fp_config), &info_fp_config,NULL);
            printf("        Double FP Config:\n");
            info_uint = info_fp_config & CL_FP_DENORM ? 1 : 0;
            printf("            Denorm: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_INF_NAN ? 1 : 0;
            printf("            Inf Nan: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_ROUND_TO_NEAREST ? 1 : 0;
            printf("            Round Nearest: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_ROUND_TO_ZERO ? 1 : 0;
            printf("            Round To Zero: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_ROUND_TO_INF ? 1 : 0;
            printf("            Round to Inf: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_FMA ? 1 : 0;
            printf("            FMA: %u\n", info_uint);
#endif
            
            printInfoCLBOOL(device[j], CL_DEVICE_ENDIAN_LITTLE,
                            "Endian Little");
            printInfoCLBOOL(device[j], CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                            "Error Correction Support");

            clGetDeviceInfo(device[j], CL_DEVICE_EXECUTION_CAPABILITIES,
                            sizeof(cl_device_exec_capabilities), 
                            &info_exec_capabilities, NULL);
            printf("        Execution Capabilities:\n");
            info_uint = info_exec_capabilities & CL_EXEC_KERNEL ? 1 : 0;
            printf("            Exec Kernel: %u\n", info_uint);
            info_uint = info_exec_capabilities & CL_EXEC_NATIVE_KERNEL ? 1 : 0;
            printf("            Exec Native Kernel: %u\n", info_uint);
           
            printInfoSTR(device[j], CL_DEVICE_EXTENSIONS, "Extensions");
            printInfoCLULONG(device[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                            "Global Mem Cache Size");

            clGetDeviceInfo(device[j], CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                            sizeof(cl_device_mem_cache_type),
                            &info_mem_cache_type, NULL);
            if(info_mem_cache_type == CL_NONE)
                printf("        Global Mem Cache Type: NONE\n");
            else if(info_mem_cache_type == CL_READ_ONLY_CACHE)
                printf("        Global Mem Cache Type: READ ONLY\n");
            else if(info_mem_cache_type == CL_READ_WRITE_CACHE)
                printf("        Global Mem Cache Type: READ/WRITE\n");
            else
                printf("        Global Mem Cache Type: ERROR\n");

            printInfoCLUINT(device[j], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                            "Global Mem Cacheline Size");
            printInfoCLULONG(device[j], CL_DEVICE_GLOBAL_MEM_SIZE,
                            "Global Mem Size");

#ifdef CL_DEVICE_HALF_FP_CONFIG
            clGetDeviceInfo(device[j], CL_DEVICE_HALF_FP_CONFIG,
                            sizeof(cl_device_fp_config), &info_fp_config,NULL);
            printf("        HALF FP Config:\n");
            info_uint = info_fp_config & CL_FP_DENORM ? 1 : 0;
            printf("            Denorm: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_INF_NAN ? 1 : 0;
            printf("            Inf Nan: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_ROUND_TO_NEAREST ? 1 : 0;
            printf("            Round Nearest: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_ROUND_TO_ZERO ? 1 : 0;
            printf("            Round To Zero: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_ROUND_TO_INF ? 1 : 0;
            printf("            Round to Inf: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_FMA ? 1 : 0;
            printf("            FMA: %u\n", info_uint);
#endif
           
            printInfoCLBOOL(device[j], CL_DEVICE_IMAGE_SUPPORT,
                            "Image Support");
            printInfoSIZET(device[j], CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                            "Image 2D Max Height");
            printInfoSIZET(device[j], CL_DEVICE_IMAGE2D_MAX_WIDTH,
                            "Image 2D Max Width");
            printInfoSIZET(device[j], CL_DEVICE_IMAGE3D_MAX_DEPTH,
                            "Image 3D Max Depth");
            printInfoSIZET(device[j], CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                            "Image 3D Max Height");
            printInfoSIZET(device[j], CL_DEVICE_IMAGE3D_MAX_WIDTH,
                            "Image 3D Max Width");

            printInfoCLULONG(device[j], CL_DEVICE_LOCAL_MEM_SIZE,
                            "Local Mem Size");

            clGetDeviceInfo(device[j], CL_DEVICE_LOCAL_MEM_TYPE,
                            sizeof(cl_device_local_mem_type),
                            &info_local_mem_type, NULL);
            if(info_local_mem_type == CL_LOCAL)
                printf("        Local Mem Type: LOCAL\n");
            else if(info_local_mem_type == CL_GLOBAL)
                printf("        Local Mem Type: GLOBAL\n");
            else
                printf("        Local Mem Type: ERROR\n");

            printInfoCLUINT(device[j], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            "Max Clock Frequency (MHz)");
            printInfoCLUINT(device[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                            "Max Compute Units");
            printInfoCLUINT(device[j], CL_DEVICE_MAX_CONSTANT_ARGS,
                            "Max Constant Args");
            printInfoCLULONG(device[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                            "Max Constant Buffer Size");
            printInfoCLULONG(device[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            "Max Mem Alloc Size");
            printInfoSIZET(device[j], CL_DEVICE_MAX_PARAMETER_SIZE,
                            "Max Parameter Size");
            printInfoCLUINT(device[j], CL_DEVICE_MAX_READ_IMAGE_ARGS,
                            "Max Read Image Args");
            printInfoCLUINT(device[j], CL_DEVICE_MAX_SAMPLERS,
                            "Max Samplers");
            printInfoSIZET(device[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            "Max Work Group Size");
            printInfoCLUINT(device[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            "Max Work Item Dimensions");

            clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(cl_uint), &info_uint, NULL);
            size_t sizes[info_uint];
            clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            info_uint*sizeof(size_t), sizes, NULL);
            int k;
            printf("        Max Work Item Sizes: (");
            for(k=0; k<info_uint-1; k++)
                printf("%zu,", sizes[k]);
            printf("%zu)\n", sizes[info_uint-1]);

            printInfoCLUINT(device[j], CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
                            "Max Write Image Args");
            printInfoCLUINT(device[j], CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                            "Mem Base Addr Align");
            printInfoCLUINT(device[j], CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
                            "Min Data Type Align Size");
            printInfoSTR(device[j], CL_DEVICE_NAME, "Name");
            printInfoCLUINT(device[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                            "Preferred Vector Width Char");
            printInfoCLUINT(device[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                            "Preferred Vector Width Short");
            printInfoCLUINT(device[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                            "Preferred Vector Width Int");
            printInfoCLUINT(device[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                            "Preferred Vector Width Long");
            printInfoCLUINT(device[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                            "Preferred Vector Width Float");
            printInfoCLUINT(device[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                            "Preferred Vector Width Double");
            printInfoSTR(device[j], CL_DEVICE_PROFILE, "Profile");
            printInfoSIZET(device[j], CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                            "Profiling Timer Resolution (ns)");

            clGetDeviceInfo(device[j], CL_DEVICE_QUEUE_PROPERTIES,
                            sizeof(cl_command_queue_properties),
                            &info_command_queue_props, NULL);
            printf("        Queue Properties:\n");
            info_uint = info_command_queue_props & 
                                CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ? 1 : 0;
            printf("            Queue Out-of-Order Exec Mode Enabled: %u\n",
                    info_uint);
            info_uint = info_command_queue_props & 
                            CL_QUEUE_PROFILING_ENABLE ? 1 : 0;
            printf("            Queue Profiling Enabled: %u\n", info_uint);
            
            clGetDeviceInfo(device[j], CL_DEVICE_SINGLE_FP_CONFIG,
                            sizeof(cl_device_fp_config), &info_fp_config,NULL);
            printf("        SINGLE FP Config:\n");
            info_uint = info_fp_config & CL_FP_DENORM ? 1 : 0;
            printf("            Denorm: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_INF_NAN ? 1 : 0;
            printf("            Inf Nan: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_ROUND_TO_NEAREST ? 1 : 0;
            printf("            Round Nearest: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_ROUND_TO_ZERO ? 1 : 0;
            printf("            Round To Zero: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_ROUND_TO_INF ? 1 : 0;
            printf("            Round to Inf: %u\n", info_uint);
            info_uint = info_fp_config & CL_FP_FMA ? 1 : 0;
            printf("            FMA: %u\n", info_uint);

            clGetDeviceInfo(device[j], CL_DEVICE_TYPE,
                            sizeof(cl_device_type),
                            &info_type, NULL);
            if(info_type == CL_DEVICE_TYPE_CPU)
                printf("        Type: CPU\n");
            else if(info_type == CL_DEVICE_TYPE_GPU)
                printf("        Type: GPU\n");
            else if(info_type == CL_DEVICE_TYPE_ACCELERATOR)
                printf("        Type: ACCELERATOR\n");
            else if(info_type == CL_DEVICE_TYPE_DEFAULT)
                printf("        Type: DEFAULT\n");
            else
                printf("        Type: ERROR\n");

            printInfoSTR(device[j], CL_DEVICE_VENDOR, "Vendor");
            printInfoCLUINT(device[j], CL_DEVICE_VENDOR_ID, "Vendor ID");
            printInfoSTR(device[j], CL_DEVICE_VERSION, "Version");
            printInfoSTR(device[j], CL_DRIVER_VERSION, "Driver Version");

            printf("\n");
        }
    }

    return 0;
}
