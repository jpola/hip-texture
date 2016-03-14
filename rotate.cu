#include <cuda_runtime.h>
#include "image_descriptor.hpp"
#include <stdio.h>

// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

template<typename T>
__global__ void transformKernel(T* outputData,
                                int width,
                                int height,
                                T theta)
{
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    T u = (T)x - (T)width/2;
    T v = (T)y - (T)height/2;
    T tu = u*cosf(theta) - v*sinf(theta);
    T tv = v*cosf(theta) + u*sinf(theta);

    tu /= (T)width;
    tv /= (T)height;

    // read from texture and write to global memory
    T val = tex2D(tex, tu+0.5f, tv+0.5f);
    if(x == 0 && y == 0)
        printf("u = %f, v = %f, tu = %f, tv = %f, val = %f\n", u, v, tu, tv, val);
    if(x == 1 && y == 0)
        printf("u = %f, v = %f, tu = %f, tv = %f, val = %f\n", u, v, tu, tv, val);
    if(x == 2 && y == 0)
        printf("u = %f, v = %f, tu = %f, tv = %f, val = %f\n", u, v, tu, tv, val);
    outputData[y*width + x] = val;
}


template<typename T>
void transform_image(T* data, const image_description<T>& img_desc, T angle)
{

    size_t size = img_desc.width * img_desc.height * sizeof(T);

    T* d_data = NULL;
    cudaMalloc((void **) &d_data, size);

    /**
     * Example, for float texels we could create a channel with
     *
     * cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
     *
     * while for short4 texels this would be
     *
     * cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSigned);
     */
    //define channel descriptor for image 32 means 32 bits per pixel = float;
    cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    //prepare array to fetch data from. It is the copy of input image
    cudaArray *cuArray;
    cudaMallocArray(&cuArray, &channelDesc, img_desc.width, img_desc.height);
    cudaMemcpyToArray(cuArray, 0, 0, data, size, cudaMemcpyHostToDevice);

    //set texture parameters
    if(img_desc.x_addr_mode == CLAMP)
    {
        tex.addressMode[0] = cudaAddressModeWrap;
        tex.addressMode[1] = cudaAddressModeClamp;
    }
    else
    {
        tex.addressMode[0] = cudaAddressModeWrap;
        tex.addressMode[1] = cudaAddressModeWrap;

    }
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTextureToArray(tex, cuArray, channelDesc);

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(img_desc.width / dimBlock.x, img_desc.height / dimBlock.y, 1);

    transformKernel<<<dimGrid, dimBlock, 0>>>(d_data,
                                              (int)img_desc.width,
                                              (int)img_desc.height,
                                              angle);

    cudaDeviceSynchronize();
    cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFreeArray(cuArray);
}

//instantiation of a template definition
// other option is to move the definition to header file
template void transform_image<float>(float* data, const image_description<float>& img_desc, float f);
