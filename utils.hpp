#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <cassert>
#include <iostream>

template<class T, class S>
inline bool
compareData(const T *reference, const T *data, const unsigned int len,
            const S epsilon, const float threshold)
{
    assert(epsilon >= 0);

    bool result = true;
    unsigned int error_count = 0;

    for (unsigned int i = 0; i < len; ++i)
    {
        float diff = (float)reference[i] - (float)data[i];
        bool comp = (diff <= epsilon) && (diff >= -epsilon);
        result &= comp;

        error_count += !comp;

//                if (! comp)
//                {
//                    std::cerr << "ERROR, i = " << i << ",\t "
//                              << reference[i] << " / "
//                              << data[i] << " "
//                              << 1.f - (reference[i] / data[i]) << " "
//                                 << " (reference / data)\n";
//                }
    }

    if (threshold == 0.0f)
    {
        return (result) ? true : false;
    }
    else
    {
        if (error_count)
        {
            printf("%4.2f(%%) of bytes mismatched (count=%d)\n", (float)error_count*100/(float)len, error_count);
        }

        return (len*threshold > error_count) ? true : false;
    }
}


//displays image line
template<typename T>
void display_img_line(const unsigned int line, T* data, const image_description<T>& img_desc)
{
    std::cout << std::endl;
    for (unsigned int col = 0; col < img_desc.width; col++)
    {
        std::cout << data[line + img_desc.width*col] << ", ";
    }
    std::cout << std::endl;
}
#endif //_UTILS_HPP_
