#include <iostream>
#include <string>
#include <CImg.h>

#include "image_descriptor.hpp"
#include "rotate.hpp"
#include "rotate_hip.hpp"
#include "utils.hpp"

#include "moving_average.hpp"

using namespace cimg_library;


void test_rotate()
{
    //change this to test other data types
    typedef float T;

    std::string image_file =
    "data/lena_bw.pgm";
    //"data/lena_small.pgm";

    CImg<T> cuda_image(image_file.c_str());
    cuda_image.normalize(0.f, 1.f);
    CImg<T> hip_image(image_file.c_str());
    hip_image.normalize(0.f, 1.f);

    T* cuda_data = cuda_image.data();
    T* hip_data  = hip_image.data();

    std::cout << "Image : " <<  image_file << "\n"
              << "widht: " << cuda_image.width() << " "
              << "height: " << cuda_image.height() << " "
              << "depth: " << cuda_image.depth() << " "
              << "spectrum: " << cuda_image.spectrum() << " "
              << "size: " <<cuda_image.size() << std::endl;

    image_description<T> img_desc;

    img_desc.height = cuda_image.height();
    img_desc.width = cuda_image.width();

    img_desc.x_addr_mode = CLAMP;
    img_desc.y_addr_mode = CLAMP;

    transform_image(cuda_data, img_desc, 0.5f);

    transform_host_image(hip_data, img_desc, 0.5f);


//    std::string ref_image_file = "data/ref_rotated.pgm";
//    CImg<T> ref_image(ref_image_file.c_str());
//    ref_image.normalize(0.f, 1.f);

    //compareData(ref_image.data(), d, img_desc.width*img_desc.height, 5e-2f, 0.15f);

    cuda_image.normalize(0, 255);
    cuda_image.save("data/lena_cuda_result.pgm");

    hip_image.normalize(0, 255);
    hip_image.save("data/lena_hip_result.pgm");

}



int main(int argc, char *argv[])
{



    calculate_moving_average();




    //test_rotate();







    return 0;
}

