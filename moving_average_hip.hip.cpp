#include <hip_runtime.h>
#include <iostream>

#include "moving_average_hip.hpp"
#include "image_descriptor.hpp"
#include "image_utils.hpp"


void moving_average_hip(float* src, float* dst, const int N, const int R)
{

}

void calculate_moving_average_hip()
{
    const int N = 20;
    const int R = 6;

    float *h_in = new float[N];
    float *h_out = new float[N];

    for (int i = 0; i < N; i++)
        h_in[i] = (float) (rand() % 10);

    moving_average_hip(h_in, h_out, N, R);

    for (int i = 0; i < N; i++)
    {
        std::cout << " HIP GPU = " << h_out[i] << std::endl;
    }

    delete [] h_in;
    delete [] h_out;
}
