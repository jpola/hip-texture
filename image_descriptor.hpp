#ifndef _IMAGE_DESCRIPTOR_HPP_
#define _IMAGE_DESCRIPTOR_HPP_

enum address_mode
{
    CLAMP,
    WRAP
};

enum filter_mode
{
    POINT,
    LINEAR
};


template<typename T>
struct image_description
{
    //boundary conditions
    address_mode x_addr_mode;
    address_mode y_addr_mode;

    //wymiary
    unsigned int height;
    unsigned int width;

    //filter mode;
    filter_mode f_mode;
};

#endif //_IMAGE_DESCRIPTOR_HPP_
