#ifndef _ROTATE_HIP_HPP_
#define _ROTATE_HIP_HPP_

template<typename T>
struct image_description;

template<typename T>
void transform_hip_image(T* data, const image_description<T>& img_desc, T angle);
template<typename T>
void transform_host_image(T* data, const image_description<T>& img_desc, T angle);

#endif // _ROTATE_HIP_HPP_
