/**
    Copyright (C) 2010  puddinpop

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 **/

#include "cudashared.h"

/**
- * Extracts a bit field from source and places the zero or sign-extended result 
- * in extract
- */
template <unsigned int BIT_START, unsigned int NUM_BITS>
struct ExtractKeyBits
{

    __device__ __forceinline__ static unsigned int Extract(const unsigned int &source)
    {
        unsigned int bits;
        //               asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "n"(BIT_START), "n"(NUM_BITS));
        unsigned int _BIT_START = BIT_START;
        unsigned int _NUM_BITS = NUM_BITS;
        asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "r"(_BIT_START), "r"(_NUM_BITS));
        return bits;
    }
};

template <unsigned int BIT_START, unsigned int NUM_BITS>
struct InsertKeyBits
{

    __device__ __forceinline__ static unsigned int Insert(const unsigned int &orig, const unsigned int &source)
    {
        unsigned int bits;
        //               asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "n"(BIT_START), "n"(NUM_BITS));
        unsigned int _BIT_START = BIT_START;
        unsigned int _NUM_BITS = NUM_BITS;
        asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(bits) : "r"(source), "r"(orig), "r"(_BIT_START), "r"(_NUM_BITS));
        return bits;
    }
};

struct Rot
{

    __device__ __forceinline__ static unsigned int rot(const unsigned int &a, const unsigned int bit)
    {
        unsigned int out;
        //               asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "n"(BIT_START), "n"(NUM_BITS));
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(a), "r"(bit));
        return out;
    }
};

//template <unsigned int SHIFT>

struct Shift
{

    __device__ __forceinline__ unsigned int operator()(const unsigned int &source, const unsigned int& _BIT_START, const unsigned int& _NUM_BITS)
    {
        unsigned int bits;
        //               asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "n"(BIT_START), "n"(NUM_BITS));
        //                 const unsigned int _BIT_START = SHIFT;
        //                 const unsigned int _NUM_BITS = 32-SHIFT;
        asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "r"(_BIT_START), "r"(_NUM_BITS));
        return bits;
    }
};

#define JOIN_DO(x,y) x##y
#define J(x,y) JOIN_DO(x,y)

#define byteswap(x) (((x>>24) & 0x000000ff) | ((x>>8) & 0x0000ff00) | ((x<<8) & 0x00ff0000) | ((x<<24) & 0xff000000))
//#define rotateright(x,bits) (((x & 0xffffffff) >> bits) | (x << (32 - bits)))
//#define rotateright(x,bits) (Rot().rot(x,bits))
#define rotateright(x,bits) (((x ) >> bits) | (x << (32 - bits)))
//#define rotateright(x,bits) ( InsertKeyBits<32-bits,bits>().Insert(ExtractKeyBits<bits,32-bits>().Extract(x),x) )

//#define R(x) (work[x] = (rotateright(work[x-2],17)^rotateright(work[x-2],19)^((work[x-2]&0xffffffff)>>10)) + work[x -  7] + (rotateright(work[x-15],7)^rotateright(work[x-15],18)^((work[x-15]&0xffffffff)>>3)) + work[x - 16])
#define R(x,x2,x7,x15,x16) (x = (rotateright(x2,17)^rotateright(x2,19)^((x2)>>10)) + x7 + (rotateright(x15,7)^rotateright(x15,18)^((x15)>>3)) + x16)
#define RL(x2,x7,x15,x16) ((rotateright(x2,17)^rotateright(x2,19)^((x2)>>10)) + x7 + (rotateright(x15,7)^rotateright(x15,18)^((x15)>>3)) + x16)


//#define R(x,x2,x7,x15,x16) (x = (rotateright(x2,17)^rotateright(x2,19)^(Shift()(x2,10,22))) + x7 + (rotateright(x15,7)^rotateright(x15,18)^(Shift()(x15,3,29))) + x16)


#define sharound(a,b,c,d,e,f,g,h,x,K) {const uint t1=(rotateright(e,6)^rotateright(e,11)^rotateright(e,25))+(g^(e&(f^g)))+K+x; const uint t2=(rotateright(a,2)^rotateright(a,13)^rotateright(a,22))+((a&b)|((a|b)&c)); d+=t1+h; h+=t1+t2;}
#define sharoundLL(a,b,c,d,e,f,g,h,x) {d+=h+(rotateright(e,6)^rotateright(e,11)^rotateright(e,25))+(g^(e&(f^g)))+x; }
#define sharoundL(a,b,c,d,e,f,g,h,K) {const uint t1=(rotateright(e,6)^rotateright(e,11)^rotateright(e,25))+(g^(e&(f^g)))+K;const uint t2=(rotateright(a,2)^rotateright(a,13)^rotateright(a,22))+((a&b)|((a|b)&c)); d+=t1+h; h+=t1+t2;}

__global__
#if __CUDA_ARCH__ == 130
__launch_bounds__(256, 2)
#elif __CUDA_ARCH__ == 200
__launch_bounds__(256, 4)
#elif __CUDA_ARCH__ == 300
__launch_bounds__(256, 8)
#elif __CUDA_ARCH__ == 350
__launch_bounds__(256, 8)
#endif
void cuda_process(uint32 __restrict__ *in, uint32 __restrict__ *out, const uint32 nonceIn, const uint32 loops, const uint32 bits)
{
#if __CUDA_ARCH__ == 130
    if (*out != 0) return;
#endif

    const uint myid = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint nonce = nonceIn + (myid << bits);

    // the first 3 rounds we can do outside the loop because they depend on work[0] through work[2] which won't change
    __shared__ uint32 s_in[cudaIn::count];

    if (threadIdx.x < cudaIn::count)
    {
        s_in[threadIdx.x] = in[threadIdx.x];
    }
    __syncthreads();


#pragma unroll 1
    for (uint32 it = 0; it < loops; ++it)
    {
        if (*out != 0) return;
        uint work3 = nonce + it;
        unsigned int A, B, C, D, E, F, G, H;

        A = s_in[cudaIn::preCals + 0] + work3;
        B = s_in[cudaIn::preCals + 1];
        C = s_in[cudaIn::preCals + 2];
        D = s_in[cudaIn::preCals + 3];
        E = s_in[cudaIn::preCals + 4] + work3;
        F = s_in[cudaIn::preCals + 5];
        G = s_in[cudaIn::preCals + 6];
        H = s_in[cudaIn::preCals + 7];

        uint work4 = 0x80000000;
        sharound(E, F, G, H, A, B, C, D, work4, 0x3956C25B);
        uint work5 = 0x00000000;
        sharound(D, E, F, G, H, A, B, C, work5, 0x59F111F1);
        uint work6 = 0x00000000;
        sharound(C, D, E, F, G, H, A, B, work6, 0x923F82A4);
        uint work7 = 0x00000000;
        sharound(B, C, D, E, F, G, H, A, work7, 0xAB1C5ED5);
        uint work8 = 0x00000000;
        sharound(A, B, C, D, E, F, G, H, work8, 0xD807AA98);
        uint work9 = 0x00000000;
        sharound(H, A, B, C, D, E, F, G, work9, 0x12835B01);
        uint work10 = 0x00000000;
        sharound(G, H, A, B, C, D, E, F, work10, 0x243185BE);
        uint work11 = 0x00000000;
        sharound(F, G, H, A, B, C, D, E, work11, 0x550C7DC3);
        uint work12 = 0x00000000;
        sharound(E, F, G, H, A, B, C, D, work12, 0x72BE5D74);
        uint work13 = 0x00000000;
        sharound(D, E, F, G, H, A, B, C, work13, 0x80DEB1FE);
        uint work14 = 0x00000000;
        sharound(C, D, E, F, G, H, A, B, work14, 0x9BDC06A7);
        uint work15 = 0x00000280;
        sharound(B, C, D, E, F, G, H, A, work15, 0xC19BF174);

        uint work0 = s_in[cudaIn::merkle];
        uint work1 = s_in[cudaIn::ntime];
        uint work16;
        sharound(A, B, C, D, E, F, G, H, R(work16, work14, work9, work1, work0), 0xE49B69C1);
        uint work2 = s_in[cudaIn::nbits];
        uint work17;
        sharound(H, A, B, C, D, E, F, G, R(work17, work15, work10, work2, work1), 0xEFBE4786);
        uint work18;
        sharound(G, H, A, B, C, D, E, F, R(work18, work16, work11, work3, work2), 0x0FC19DC6);
        uint work19;
        sharound(F, G, H, A, B, C, D, E, R(work19, work17, work12, work4, work3), 0x240CA1CC);
        uint work20;
        sharound(E, F, G, H, A, B, C, D, R(work20, work18, work13, work5, work4), 0x2DE92C6F);
        uint work21;
        sharound(D, E, F, G, H, A, B, C, R(work21, work19, work14, work6, work5), 0x4A7484AA);
        uint work22;
        sharound(C, D, E, F, G, H, A, B, R(work22, work20, work15, work7, work6), 0x5CB0A9DC);
        uint work23;
        sharound(B, C, D, E, F, G, H, A, R(work23, work21, work16, work8, work7), 0x76F988DA);
        uint work24;
        sharound(A, B, C, D, E, F, G, H, R(work24, work22, work17, work9, work8), 0x983E5152);
        uint work25;
        sharound(H, A, B, C, D, E, F, G, R(work25, work23, work18, work10, work9), 0xA831C66D);
        uint work26;
        sharound(G, H, A, B, C, D, E, F, R(work26, work24, work19, work11, work10), 0xB00327C8);
        uint work27;
        sharound(F, G, H, A, B, C, D, E, R(work27, work25, work20, work12, work11), 0xBF597FC7);
        uint work28;
        sharound(E, F, G, H, A, B, C, D, R(work28, work26, work21, work13, work12), 0xC6E00BF3);
        uint work29;
        sharound(D, E, F, G, H, A, B, C, R(work29, work27, work22, work14, work13), 0xD5A79147);
        uint work30;
        sharound(C, D, E, F, G, H, A, B, R(work30, work28, work23, work15, work14), 0x06CA6351);
        uint work31;
        sharound(B, C, D, E, F, G, H, A, R(work31, work29, work24, work16, work15), 0x14292967);
        uint work32;
        sharound(A, B, C, D, E, F, G, H, R(work32, work30, work25, work17, work16), 0x27B70A85);
        uint work33;
        sharound(H, A, B, C, D, E, F, G, R(work33, work31, work26, work18, work17), 0x2E1B2138);
        uint work34;
        sharound(G, H, A, B, C, D, E, F, R(work34, work32, work27, work19, work18), 0x4D2C6DFC);
        uint work35;
        sharound(F, G, H, A, B, C, D, E, R(work35, work33, work28, work20, work19), 0x53380D13);
        uint work36;
        sharound(E, F, G, H, A, B, C, D, R(work36, work34, work29, work21, work20), 0x650A7354);
        uint work37;
        sharound(D, E, F, G, H, A, B, C, R(work37, work35, work30, work22, work21), 0x766A0ABB);
        uint work38;
        sharound(C, D, E, F, G, H, A, B, R(work38, work36, work31, work23, work22), 0x81C2C92E);
        uint work39;
        sharound(B, C, D, E, F, G, H, A, R(work39, work37, work32, work24, work23), 0x92722C85);
        uint work40;
        sharound(A, B, C, D, E, F, G, H, R(work40, work38, work33, work25, work24), 0xA2BFE8A1);
        uint work41;
        sharound(H, A, B, C, D, E, F, G, R(work41, work39, work34, work26, work25), 0xA81A664B);
        uint work42;
        sharound(G, H, A, B, C, D, E, F, R(work42, work40, work35, work27, work26), 0xC24B8B70);
        uint work43;
        sharound(F, G, H, A, B, C, D, E, R(work43, work41, work36, work28, work27), 0xC76C51A3);
        uint work44;
        sharound(E, F, G, H, A, B, C, D, R(work44, work42, work37, work29, work28), 0xD192E819);
        uint work45;
        sharound(D, E, F, G, H, A, B, C, R(work45, work43, work38, work30, work29), 0xD6990624);
        uint work46;
        sharound(C, D, E, F, G, H, A, B, R(work46, work44, work39, work31, work30), 0xF40E3585);
        uint work47;
        sharound(B, C, D, E, F, G, H, A, R(work47, work45, work40, work32, work31), 0x106AA070);
        uint work48;
        sharound(A, B, C, D, E, F, G, H, R(work48, work46, work41, work33, work32), 0x19A4C116);
        uint work49;
        sharound(H, A, B, C, D, E, F, G, R(work49, work47, work42, work34, work33), 0x1E376C08);
        uint work50;
        sharound(G, H, A, B, C, D, E, F, R(work50, work48, work43, work35, work34), 0x2748774C);
        uint work51;
        sharound(F, G, H, A, B, C, D, E, R(work51, work49, work44, work36, work35), 0x34B0BCB5);
        uint work52;
        sharound(E, F, G, H, A, B, C, D, R(work52, work50, work45, work37, work36), 0x391C0CB3);
        uint work53;
        sharound(D, E, F, G, H, A, B, C, R(work53, work51, work46, work38, work37), 0x4ED8AA4A);
        uint work54;
        sharound(C, D, E, F, G, H, A, B, R(work54, work52, work47, work39, work38), 0x5B9CCA4F);
        uint work55;
        ;
        sharound(B, C, D, E, F, G, H, A, R(work55, work53, work48, work40, work39), 0x682E6FF3);
        uint work56;
        sharound(A, B, C, D, E, F, G, H, R(work56, work54, work49, work41, work40), 0x748F82EE);
        uint work57;
        sharound(H, A, B, C, D, E, F, G, R(work57, work55, work50, work42, work41), 0x78A5636F);
        uint work58;
        sharound(G, H, A, B, C, D, E, F, R(work58, work56, work51, work43, work42), 0x84C87814);
        uint work59;
        sharound(F, G, H, A, B, C, D, E, R(work59, work57, work52, work44, work43), 0x8CC70208);
        uint work60;
        sharound(E, F, G, H, A, B, C, D, R(work60, work58, work53, work45, work44), 0x90BEFFFA);
        uint work61;
        sharound(D, E, F, G, H, A, B, C, R(work61, work59, work54, work46, work45), 0xA4506CEB);
        //uint work62;
        sharound(C, D, E, F, G, H, A, B, RL(work60, work55, work47, work46), 0xBEF9A3F7);
        //uint work63;
        sharound(B, C, D, E, F, G, H, A, RL(work61, work56, work48, work47), 0xC67178F2);

        // hash the hash now
        uint work0x = s_in[ 0] + A;
        uint work1x = s_in[ 1] + B;
        uint work2x = s_in[2] + C;
        uint work3x = s_in[ 3] + D;
        uint work4x = s_in[ 4] + E;
        uint work5x = s_in[ 5] + F;
        uint work6x = s_in[ 6] + G;
        uint work7x = s_in[ 7] + H;

        A = 0x6a09e667;
        B = 0xbb67ae85;
        C = 0x3c6ef372;
        D = 0xa54ff53a;
        E = 0x510e527f;
        F = 0x9b05688c;
        G = 0x1f83d9ab;
        H = 0x5be0cd19;

        sharound(A, B, C, D, E, F, G, H, work0x, 0x428A2F98);
        sharound(H, A, B, C, D, E, F, G, work1x, 0x71374491);
        sharound(G, H, A, B, C, D, E, F, work2x, 0xB5C0FBCF);


        sharound(F, G, H, A, B, C, D, E, work3x, 0xE9B5DBA5);
        sharound(E, F, G, H, A, B, C, D, work4x, 0x3956C25B);
        sharound(D, E, F, G, H, A, B, C, work5x, 0x59F111F1);
        sharound(C, D, E, F, G, H, A, B, work6x, 0x923F82A4);
        sharound(B, C, D, E, F, G, H, A, work7x, 0xAB1C5ED5);
        uint work8x = 0x80000000;
        sharound(A, B, C, D, E, F, G, H, work8x, 0xD807AA98);
        uint work9x = 0x00000000;
        sharound(H, A, B, C, D, E, F, G, work9x, 0x12835B01);
        uint work10x = 0x00000000;
        sharound(G, H, A, B, C, D, E, F, work10x, 0x243185BE);
        uint work11x = 0x00000000;
        sharound(F, G, H, A, B, C, D, E, work11x, 0x550C7DC3);
        uint work12x = 0x00000000;
        sharound(E, F, G, H, A, B, C, D, work12x, 0x72BE5D74);
        uint work13x = 0x00000000;
        sharound(D, E, F, G, H, A, B, C, work13x, 0x80DEB1FE);
        uint work14x = 0x00000000;
        sharound(C, D, E, F, G, H, A, B, work14x, 0x9BDC06A7);
        uint work15x = 0x00000100;
        sharound(B, C, D, E, F, G, H, A, work15x, 0xC19BF174);

        sharound(A, B, C, D, E, F, G, H, R(work16, work14x, work9x, work1x, work0x), 0xE49B69C1);
        sharound(H, A, B, C, D, E, F, G, R(work17, work15x, work10x, work2x, work1x), 0xEFBE4786);
        sharound(G, H, A, B, C, D, E, F, R(work18, work16, work11x, work3x, work2x), 0x0FC19DC6);
        sharound(F, G, H, A, B, C, D, E, R(work19, work17, work12x, work4x, work3x), 0x240CA1CC);
        sharound(E, F, G, H, A, B, C, D, R(work20, work18, work13x, work5x, work4x), 0x2DE92C6F);
        sharound(D, E, F, G, H, A, B, C, R(work21, work19, work14x, work6x, work5x), 0x4A7484AA);
        sharound(C, D, E, F, G, H, A, B, R(work22, work20, work15x, work7x, work6x), 0x5CB0A9DC);
        sharound(B, C, D, E, F, G, H, A, R(work23, work21, work16, work8x, work7x), 0x76F988DA);
        sharound(A, B, C, D, E, F, G, H, R(work24, work22, work17, work9x, work8x), 0x983E5152);
        sharound(H, A, B, C, D, E, F, G, R(work25, work23, work18, work10x, work9x), 0xA831C66D);
        sharound(G, H, A, B, C, D, E, F, R(work26, work24, work19, work11x, work10x), 0xB00327C8);
        sharound(F, G, H, A, B, C, D, E, R(work27, work25, work20, work12x, work11x), 0xBF597FC7);
        sharound(E, F, G, H, A, B, C, D, R(work28, work26, work21, work13x, work12x), 0xC6E00BF3);
        sharound(D, E, F, G, H, A, B, C, R(work29, work27, work22, work14x, work13x), 0xD5A79147);
        sharound(C, D, E, F, G, H, A, B, R(work30, work28, work23, work15x, work14x), 0x06CA6351);
        sharound(B, C, D, E, F, G, H, A, R(work31, work29, work24, work16, work15x), 0x14292967);
        sharound(A, B, C, D, E, F, G, H, R(work32, work30, work25, work17, work16), 0x27B70A85);
        sharound(H, A, B, C, D, E, F, G, R(work33, work31, work26, work18, work17), 0x2E1B2138);
        sharound(G, H, A, B, C, D, E, F, R(work34, work32, work27, work19, work18), 0x4D2C6DFC);
        sharound(F, G, H, A, B, C, D, E, R(work35, work33, work28, work20, work19), 0x53380D13);
        sharound(E, F, G, H, A, B, C, D, R(work36, work34, work29, work21, work20), 0x650A7354);
        sharound(D, E, F, G, H, A, B, C, R(work37, work35, work30, work22, work21), 0x766A0ABB);
        sharound(C, D, E, F, G, H, A, B, R(work38, work36, work31, work23, work22), 0x81C2C92E);
        sharound(B, C, D, E, F, G, H, A, R(work39, work37, work32, work24, work23), 0x92722C85);

        sharound(A, B, C, D, E, F, G, H, R(work40, work38, work33, work25, work24), 0xA2BFE8A1);
        sharound(H, A, B, C, D, E, F, G, R(work41, work39, work34, work26, work25), 0xA81A664B);
        sharound(G, H, A, B, C, D, E, F, R(work42, work40, work35, work27, work26), 0xC24B8B70);
        sharound(F, G, H, A, B, C, D, E, R(work43, work41, work36, work28, work27), 0xC76C51A3);
        sharound(E, F, G, H, A, B, C, D, R(work44, work42, work37, work29, work28), 0xD192E819);
        sharound(D, E, F, G, H, A, B, C, R(work45, work43, work38, work30, work29), 0xD6990624);
        sharound(C, D, E, F, G, H, A, B, R(work46, work44, work39, work31, work30), 0xF40E3585);
        sharound(B, C, D, E, F, G, H, A, R(work47, work45, work40, work32, work31), 0x106AA070);
        sharound(A, B, C, D, E, F, G, H, R(work48, work46, work41, work33, work32), 0x19A4C116);
        sharound(H, A, B, C, D, E, F, G, R(work49, work47, work42, work34, work33), 0x1E376C08);
        sharound(G, H, A, B, C, D, E, F, R(work50, work48, work43, work35, work34), 0x2748774C);
        sharound(F, G, H, A, B, C, D, E, R(work51, work49, work44, work36, work35), 0x34B0BCB5);
        sharound(E, F, G, H, A, B, C, D, R(work52, work50, work45, work37, work36), 0x391C0CB3);
        sharound(D, E, F, G, H, A, B, C, R(work53, work51, work46, work38, work37), 0x4ED8AA4A);
        sharound(C, D, E, F, G, H, A, B, R(work54, work52, work47, work39, work38), 0x5B9CCA4F);
        sharound(B, C, D, E, F, G, H, A, R(work55, work53, work48, work40, work39), 0x682E6FF3);
        sharound(A, B, C, D, E, F, G, H, R(work56, work54, work49, work41, work40), 0x748F82EE);
        sharound(H, A, B, C, D, E, F, G, R(work57, work55, work50, work42, work41), 0x78A5636F);
        sharound(G, H, A, B, C, D, E, F, R(work58, work56, work51, work43, work42), 0x84C87814);
        sharound(F, G, H, A, B, C, D, E, RL(work57, work52, work44, work43), 0x8CC70208);
        sharoundLL(E, F, G, H, A, B, C, D, RL(work58, work53, work45, work44));
        //sharound(D,E,F,G,H,A,B,C,R(work61,work59,work54,work46,work45),0xA4506CEB);

        //we don't need to do these last 2 rounds as they update F, B, E and A, but we only care about G and H
        //sharound(C,D,E,F,G,H,A,B,R(62),0xBEF9A3F7);
        //sharound(B,C,D,E,F,G,H,A,R(63),0xC67178F2);

        if ((-(0x5be0cd19 + 0x90BEFFFA) == H))// && (G<=bestg))
        {
            //atomicExch(out, nonce + it);
            *out=nonce+it;
        }

    }


}

__global__ void setValue(uint32* ptr)
{
    *ptr = 0u;
}
//#include <iostream>

void bitcoinSearch(cudaStream_t stream, int grid, int threads, uint32 *in, uint32 *out, const uint32 nonce, const int unsigned loops, const unsigned int bits)
{
    //    std::cout << "stream=" << (int*) stream << " grid" << grid << " th=" << threads << " none=" << nonce << " bits=" << bits << std::endl;
    cuda_process << <grid, threads, 0, stream >> >(in, out, nonce, loops, bits);
}

void setToZero(cudaStream_t stream, uint32* ptr)
{
    setValue << <1, 1, 0, stream >> >(ptr);
}