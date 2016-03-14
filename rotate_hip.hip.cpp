#include <hip_runtime.h>
#include <iostream>
#include "image_descriptor.hpp"
#include "image_utils.hpp"
#include <stdio.h>

template<typename T, typename addresser>
void transform_host_kernel(T* src, T* dst, int width, int height, T theta, addresser f)
{
    std::cout << "Host kernel call" << std::endl;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            T u = (T)x - (T)width/2;
            T v = (T)y - (T)height/2;

            T tu = u*cosf(theta) - v*sinf(theta);
            T tv = v*cosf(theta) + u*sinf(theta);

            tu /= (T)width;
            tv /= (T)height;

            T val = get(src, tu + 0.5f, tv + 0.5f, width, height, f);

            if (x == 0 && y == 0)
                printf("u = %f, v = %f, tu = %f, tv = %f, val = %f\n", u, v, tu, tv, val);
            if (x == 1 && y == 0)
                printf("u = %f, v = %f, tu = %f, tv = %f, val = %f\n", u, v, tu, tv, val);
            if (x == 2 && y == 0)
                printf("u = %f, v = %f, tu = %f, tv = %f, val = %f\n", u, v, tu, tv, val);

            dst[y*width + x ] = val;


        }
}

template<typename T, typename addresser>
__global__ void transformHIPKernel(hipLaunchParm lp,
                                   T* src,
                                   T* dst,
                                   int width,
                                   int height,
                                   T theta,
                                   addresser f)
{
    // calculate normalized texture coordinates
    printf("CALL FROM HIP KERNEL\n");
    unsigned int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    unsigned int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    T u = (T)x - (T)width/2;
    T v = (T)y - (T)height/2;
    T tu = u*cosf(theta) - v*sinf(theta);
    T tv = v*cosf(theta) + u*sinf(theta);

    tu /= (T)width;
    tv /= (T)height;

    //    printf("pixel coord (%f, %f)\n", tu, tv);
    dst[x + width*y] = get(src, tu + 0.5f, tv + 0.5f, width, height, f); ////src[x + width*y];//

}

template<typename T>
void transform_host_image(T* src, const image_description<T>& img_desc, T angle)
{

    size_t size = img_desc.width*img_desc.height;
    T* dst = new T[size];

    if (img_desc.x_addr_mode == WRAP)
        transform_host_kernel(src, dst, img_desc.width, img_desc.height, angle, wrap<int>);
    else
        transform_host_kernel(src, dst, img_desc.width, img_desc.height, angle, clamp<int>);


    //std::copy(src, src+size, dst);
    std::copy(dst, dst + size, src);

    delete [] dst;


    //    else
    //        hipLaunchKernel(HIP_KERNEL_NAME(transformHIPKernel),
    //                        dim3(dimGrid), dim3(dimBlock), 0, 0,
    //                        d_src, d_dst, (int)img_desc.width, (int)img_desc.height, angle,
    //                        clamp<int>);


}

template <typename T>
void transform_hip_image(T* src, const image_description<T>& img_desc, T angle)
{
    size_t size = img_desc.width * img_desc.height * sizeof(T);
    std::cout << "starting HIP" << std::endl;

    T* d_src = NULL;
    T* d_dst = NULL;

    hipError_t hipError = hipErrorUnknown;

    hipError = hipMalloc((void **) &d_src, size);
    if(hipError != hipSuccess || d_src == 0)
    {
        std::cerr << "Error: malloc d_src" << std::endl;
    }
    else
    {
        std::cerr << hipGetErrorString(hipError) << std::endl;
    }
    hipError = hipMalloc((void **) &d_dst, size);
    if(hipError != hipSuccess || d_dst == 0)
    {
        std::cerr << "Error: malloc d_dst"<< hipGetErrorString(hipError) << std::endl;
    }

    //move data to gpu
    hipError = hipMemcpy(d_src, src, size, hipMemcpyHostToDevice);
    if(hipError != hipSuccess)
    {
        std::cerr << "Error: cpy src -> d_src"<< hipGetErrorString(hipError) << std::endl;
    }


    //call kernel
    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(img_desc.width / dimBlock.x, img_desc.height / dimBlock.y, 1);

    if (img_desc.x_addr_mode == WRAP)
        hipLaunchKernel(HIP_KERNEL_NAME(transformHIPKernel),
                        dim3(dimGrid), dim3(dimBlock), 0, 0,
                        d_src, d_dst, (int)img_desc.width, (int)img_desc.height, angle,
                        wrap<int>);

    //    else
    //        hipLaunchKernel(HIP_KERNEL_NAME(transformHIPKernel),
    //                        dim3(dimGrid), dim3(dimBlock), 0, 0,
    //                        d_src, d_dst, (int)img_desc.width, (int)img_desc.height, angle,
    //                        clamp<int>);

    hipError = hipDeviceSynchronize();
    if(hipError != hipSuccess)
    {
        std::cerr << "Error: kernel call: " << hipGetErrorString(hipError) << std::endl;
    }

    //get back the data to host
    hipError = hipMemcpy(src, d_dst, size, hipMemcpyDeviceToHost);
    if(hipError != hipSuccess)
    {
        std::cerr << "Error: cpy d_dst -> src: " << hipGetErrorString(hipError) << std::endl;
    }
    //release allocated resources
    hipFree(d_src);
    hipFree(d_dst);
}


template void transform_hip_image<float>(float* data, const image_description<float>& img_desc, float angle);
template void transform_host_image<float>(float* data, const image_description<float>& img_desc, float angle);
