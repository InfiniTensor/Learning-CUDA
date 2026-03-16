//辅助类定义
#pragma once

#include <utils.h>
#include <cstdint>

//辅助类定义,四字节对齐
class __align__(4) myPixel{
private:
    uint8_t _R;
    uint8_t _G;
    uint8_t _B;
    uint8_t _A;
public:
    __host__ __device__
    myPixel() : _R(0), _G(0), _B(0), _A(255) {}//默认初始化

    __host__ __device__
    myPixel(uint8_t R, uint8_t G, uint8_t B): _R(R), _G(G), _B(B), _A(255){}//显式初始化

    __host__ __device__
    myPixel(uint8_t common): _R(common), _G(common), _B(common), _A(common){}//显式初始化

    __host__ __device__
    myPixel(const myPixel& other){//拷贝初始化
        _R = other._R;
        _G = other._G;
        _B = other._B;
        _A = other._A;
        //*(uint32_t*)this = *(const uint32_t*)&other;
    }

    __host__ __device__
    ~myPixel(){}//析构

    __host__ __device__
    myPixel& operator=(const myPixel& other){//拷贝赋值
        if(this != &other){
            _R = other._R;
            _G = other._G;
            _B = other._B;
            _A = other._A;
            //*(uint32_t*)this = *(const uint32_t*)&other;
        }
        return *this;
    }

    __host__ __device__
    uint8_t R() const { return _R; }
    
    __host__ __device__
    uint8_t G() const { return _G; }

    __host__ __device__
    uint8_t B() const { return _B; }

#if defined(PLATFORM_METAX) || defined(PLATFORM_ILUVATAR)
    __host__ __device__
    int operator-(const myPixel& other) const{
        int r_d = (int)_R - (int)other._R;
        int g_d = (int)_G - (int)other._G;
        int b_d = (int)_B - (int)other._B;
        //float sum = fabsf(r_d) + fabsf(g_d) + fabsf(b_d);
        return abs(r_d) + abs(g_d) + abs(b_d);
    }
#else
    __device__
    int operator-(const myPixel& other) const {
        unsigned int p1, p2;
        p1 = *(unsigned int*)this; p2 = *(unsigned int*)&other;
        
        //莫名其妙的bug: 5090不支持__vabsdiffu4等__v*的SIMD函数,直接将两个操作数操作而非先拆分后操作? 只能采用PTX代替
        unsigned int result;
        asm("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(result) : "r"(p1), "r"(p2), "r"(0));
        return result;
    }
#endif
    
};
