#include <cuda_runtime.h>
#include <iostream>
#include "moving_average.hpp"
#include "image_utils.hpp"


void moving_average_cpu(float* dst, float* src, const int N, const int R)
{
    for(int i = 0; i < N; i++)
    {
        float average = 0.f;
        for(int k = -R; k <= R; k++)
        {
            int index = i - k;
            index = wrap<int>(index, N-1, 0);

            average = average + src[index];
        }

        dst[i] = average / (2.f * (float)R + 1.f);
    }


}


texture<float, 1, cudaReadModeElementType> tex;
__global__ void moving_average_kernel(float* __restrict__ dst, const int N, const int R)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {

        float average = 0.f;

        for (int k = -R; k <= R; k++) {
            average = average + tex1D(tex, (float)(tid - k + 0.5f)/(float)N);
        }

        dst[tid] = average / (2.f * (float)R + 1.f);
    }
}

void moving_average_gpu(float* dst, float* src, const int N, const int R)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, N, 1);
    cudaMemcpyToArray(cuArray, 0, 0, src, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaBindTextureToArray(tex, cuArray);

    tex.normalized=true;
    //only with normalized!
    tex.addressMode[0] = cudaAddressModeWrap;

    float* device_result;
    cudaMalloc((void**)&device_result, N * sizeof(float));

    moving_average_kernel<<<iDivUp(N, 256), 256>>>(device_result, N, R);

    cudaError err = cudaDeviceSynchronize();
    std::cout << "Kernel execution : " << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(dst, device_result, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "To device : " << cudaGetErrorString(err) << std::endl;
}




void calculate_moving_average()
{
    const int N = 20;
    const int R = 6;

    float *h_in = new float[N];
    float *h_out = new float[N];
    float *g_out = new float[N];

    for (int i = 0; i < N; i++)
        h_in[i] = (float) (rand() % 10);


    moving_average_cpu(h_out, h_in, N, R);
    moving_average_gpu(g_out, h_in, N, R);

    for (int i = 0; i < N; i++)
    {
        std::cout << " CPU = " << h_out[i] << " GPU = " << g_out[i] << std::endl;
    }

    delete [] h_in;
    delete [] h_out;
    delete [] g_out;



}
