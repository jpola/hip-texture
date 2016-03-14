#ifndef _ROTATE_HPP_
#define _ROTATE_HPP_

//#include "image_descriptor.hpp"
template<typename T>
struct image_description;

template<typename T>
void transform_image(T* data, const image_description<T>& img_desc, T angle);

#endif //_ROTATE_HPP_
