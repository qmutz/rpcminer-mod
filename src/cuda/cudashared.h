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

#pragma once
#include <builtin_types.h>
#include <cuda_runtime.h>
#include <cuda.h>

typedef unsigned int uint32;
typedef uint32 uint;

struct cudaIn
{

    enum
    {
        preCals = 8u, merkle = 16u, ntime = 17u, nbits = 18u
    };

    static const uint32 count = 19u;
    static const uint32 maxNonce=0x3FFFFFFF;
};

//void cuda_process_helper(cuda_in *in, cuda_out *out, const unsigned int loops, const unsigned int bits, const int grid, const int threads);

void setToZero(cudaStream_t stream, uint32* ptr);
void bitcoinSearch(cudaStream_t stream, int grid, int threads, uint32 *in, uint32 *out, const uint32 nonce, const int unsigned loops, const unsigned int bits);
