
#ifdef USE_DOUBLE
__kernel void kern_feval(__global const double *x, __global double *f)
{

    int i = get_global_id(0);
    f[i] = x[i]*x[i] - 3*x[i] + 1.0;
}
#endif

__kernel void kern_feval_f(__global const float *x, __global float *f)
{

    int i = get_global_id(0);
    f[i] = x[i]*x[i] - 3.0f*x[i] + 1.0f;
}
