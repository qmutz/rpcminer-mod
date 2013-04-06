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

#ifndef _bitcoin_miner_cuda_
#define _bitcoin_miner_cuda_

#include "../gpucommon/gpurunner.h"
#include "cudashared.h"

#include <cuda.h>


/**
 * Captures CUDA errors and prints messages to stdout, including line number and file.
 *
 * @param cmd command with cudaError_t return value to check
 */
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){std::cerr<<"<"<<__FILE__<<">:"<<__LINE__<<std::endl; throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error)));}}

#define __delete(function,var) if((var)) { CUDA_CHECK(function(var)); var=NULL; }

#define __blocksPerSM(version)  \
        ((version)<=130?2:             \
        ((version)<=210?3:             \
        ((version)<=300?5:             \
        ((version)<=350?6:1 \
        ))))

class CUDARunner:public GPURunner<unsigned long,int>
{
public:
	CUDARunner();
	~CUDARunner();

	void FindBestConfiguration();

	const unsigned long RunStep(uint32 nonce);

	uint32* GetIn()		{ return m_inH; }

private:
	void DeallocateResources();
	const bool AllocateResources(const int numb, const int numt);

	uint32* m_inH;
        uint32* m_inD;

	uint32* m_outH;
        uint32* m_outD; 
        
        cudaStream_t stream;
};



#endif	// _bitcoin_miner_cuda_
