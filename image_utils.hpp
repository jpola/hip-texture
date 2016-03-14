#ifndef _IMAGE_UTILS_HPP_
#define _IMAGE_UTILS_HPP_

inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

template<typename T>
__host__ __device__ T wrap( T i, const T upper,  const T lower = 0)
{
    int range = upper - lower + 1;
    i = (i - lower) % range;
    if (i < 0)
        return upper + 1 + i;
    else
        return lower + i;
}

template<typename T>
__host__  __device__ T clamp (const T i, const T upper, const T lower = 0)
{
    return min(max(i, lower), upper);
}

__host__ __device__ inline float frac(float x)
{
        float frac, tmp = x - (float)(int)(x);
        float frac256 = (float)(int)( tmp*256.0f + 0.5f );
        frac = frac256 / 256.0f;
        return frac;

//    float frac_part, int_part;
//    frac_part = modf(x, &int_part);
//    return frac_part;
}

__host__ __device__ inline int get_coord(int i, int j, const int stride)
{
    return j*stride + i;
}

//TODO:: add addressing mode to c
template<typename T, typename I, typename addresser>
__host__ __device__ T get(const T *data,
                 const float x, const float y,
                 const I width, const I height,  addresser f)
{

    float xb = x;// - 0.5f;
    float yb = y;// - 0.5f;

    float alpha = frac(xb);
    float beta  = frac(yb);

    I i  = f( (I)floorf(xb), width-1, 0);
    I ip = f( i + 1.f, width-1, 0);

    I j  = f((I)floorf(yb), height-1, 0);
    I jp = f(j + 1.f, height-1, 0);

    //1st option
    //2D: tex(x,y)=(1−α)(1−β)T[i,j]+
    //              α(1−β)   T[i+1,j]+
    //              (1−α)β   T[i,j+1]+
    //               αβ      T[i+1,j+1]

    //second option
    //T[i,j]
    //+ frac(α)(T[i+1,j]-T[i,j])
    //+ frac(β)(T[i,j+1]-T[i,j])
    //+ frac(αβ)(T[i,j]+T[i+1,j+1] - T[i+1, j]-T[i,j+1])

    float ab = frac(alpha*beta);

    int index00 = get_coord(i, j, width);
    int index10 = get_coord(ip, j, width);
    int index01 = get_coord(i, jp, width);
    int index11 = get_coord(ip, jp, width);


    T v = data[index00]
            + alpha * (data[index10] - data[index00])
            + beta  * (data[index01] - data[index00])
            + alpha*beta  *  (data[index00] + data[index11]- data[index10] + data[index01]);

//    T v = (1.f - alpha) * (1.f - beta)  * data[get_coord(i,  j,  width)] +
//            alpha * (1.f - beta)        * data[get_coord(ip, j,  width)] +
//           (1.f - alpha) * beta         * data[get_coord(i,  jp, width)] +
//            alpha * beta                * data[get_coord(ip, jp, width)];

    return v;
}



#endif //_IMAGE_UTILS_HPP_
