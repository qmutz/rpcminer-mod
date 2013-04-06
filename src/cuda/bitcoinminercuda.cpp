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

#define NOMINMAX

#include "bitcoinminercuda.h"
#include "cudashared.h"
#include "../cryptopp/sha.h"	// for CryptoPP::ByteReverse
//#include <cutil_inline.h>
#include <limits>

#define ALIGN_UP(offset, alignment) \
	(offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

CUDARunner::CUDARunner() : GPURunner<unsigned long, int>(TYPE_CUDA),
m_inH(NULL),
m_inD(NULL),
m_outH(NULL),
m_outD(NULL)
{

    int num_gpus = 0; //count of gpus
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    //##ERROR handling
    if (num_gpus < 1) //check if cuda device ist found
    {
        throw std::runtime_error("no CUDA capable devices detected");
    }
    m_devicecount = num_gpus;

    std::cout << num_gpus << " GPU GUDA device(s) found" << std::endl;

    if (num_gpus < m_deviceindex) //check if i can select device with diviceNumber
    {
        std::cerr << "no CUDA device " << m_deviceindex << ", only " << num_gpus << " devices found" << std::endl;
        throw std::runtime_error("CUDA capable devices can't be selected");
    }

    CUDA_CHECK(cudaSetDevice(m_deviceindex));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    std::cout << "CUDA initialized" << std::endl;
}

CUDARunner::~CUDARunner()
{
    DeallocateResources();
    CUDA_CHECK(cudaDeviceReset());
}

const bool CUDARunner::AllocateResources(const int numb, const int numt)
{
    bool allocated = true;

    CUDA_CHECK(cudaMallocHost(&m_inH, cudaIn::count * sizeof (uint32)));
    CUDA_CHECK(cudaMalloc(&m_inD, cudaIn::count * sizeof (uint32)));
    CUDA_CHECK(cudaMallocHost(&m_outH, sizeof (uint32)));
    CUDA_CHECK(cudaMalloc(&m_outD, sizeof (uint32)));

    CUDA_CHECK(cudaStreamCreate(&stream));

    printf("Done allocating CUDA resources for (%d,%d)\n", numb, numt);
    return allocated;
}

void CUDARunner::DeallocateResources()
{
    __delete(cudaFreeHost, m_inH);
    __delete(cudaFree, m_inD);
    __delete(cudaFreeHost, m_outH);
    __delete(cudaFree, m_outD);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void CUDARunner::FindBestConfiguration()
{



    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, m_deviceindex));
    std::cout << "Search Configuration for gpu named: " << prop.name << std::endl;

    int maxGridSizeX = prop.maxGridSize[0];
    int multiProcCount = prop.multiProcessorCount;
    int sm_code_verion = prop.major * 100;
    sm_code_verion += (prop.minor * 10);
    int blocksPerSM = __blocksPerSM(sm_code_verion);
    uint32 maxNonce = cudaIn::maxNonce;
    std::cout << "Your device: " << std::endl;
    std::cout << "   - " << multiProcCount << " streaming multiprocessors (SM)" << std::endl;
    std::cout << "   - " << "can run sm version " << sm_code_verion << std::endl;
    std::cout << "   - " << "bitcoin miner is optimized for " << blocksPerSM << " gpugrids per SM" << std::endl;
    std::cout << "   - " << "max value for gpugrid parameter is " << maxGridSizeX << std::endl;


    printf("max nonce %X\n", maxNonce);

    if (m_requestedgrid == -1 || m_requestedgrid > maxGridSizeX)
    {
        //auto tune
        std::cout << "Autotuning is on because no gridsize is given" << std::endl;
        int blocks = double(maxNonce / 256 / GetStepIterations()) / double(multiProcCount) / double(blocksPerSM);
        blocks *= multiProcCount*blocksPerSM;
        if (blocks == 0)
            std::cerr << "ERROR: We need a bigger maxNonce in cudashared.h" << std::endl;
        m_requestedgrid = blocks;
    }
    if (m_requestedgrid > maxGridSizeX)
    {
        std::cerr << "To big grid size fixed!!" << std::endl;
        m_requestedgrid = maxGridSizeX;
    }
    m_numb = m_requestedgrid;
    m_numt = m_requestedthreads;
    AllocateResources(m_numb, m_numt);

}

#define rotaterightH(x,bits) (((x ) >> bits) | (x << (32 - bits)))


#define sharoundH(a,b,c,d,e,f,g,h,x,K) {t1=h+(rotaterightH(e,6)^rotaterightH(e,11)^rotaterightH(e,25))+(g^(e&(f^g)))+K+x; t2=(rotaterightH(a,2)^rotaterightH(a,13)^rotaterightH(a,22))+((a&b)|((a|b)&c)); d+=t1; h=t1+t2;}
#define sharoundLH(a,b,c,d,e,f,g,h,K) {t1=h+(rotaterightH(e,6)^rotaterightH(e,11)^rotaterightH(e,25))+(g^(e&(f^g)))+K; t2=(rotaterightH(a,2)^rotaterightH(a,13)^rotaterightH(a,22))+((a&b)|((a|b)&c)); d+=t1; h=t1+t2;}

const unsigned long CUDARunner::RunStep(uint32 nonce)
{

    //clear last output
    setToZero(stream, m_outD);


    static unsigned int lastTime = 0;
    static unsigned int lastMerkle = 0;
    if (m_inH[cudaIn::ntime] != lastTime || m_inH[cudaIn::merkle] != lastMerkle)
    {
/*        lastTime = m_inH[cudaIn::ntime];
        lastMerkle = m_inH[cudaIn::merkle];
        unsigned int AH[8];
        unsigned int t1;
        unsigned int t2;
        for (int i = 0; i < 8; ++i)
            AH[i] = m_inH[i];
        sharoundH(AH[0], AH[1], AH[2], AH[3], AH[4], AH[5], AH[6], AH[7], m_inH[cudaIn::merkle], 0x428A2F98);
        sharoundH(AH[7], AH[0], AH[1], AH[2], AH[3], AH[4], AH[5], AH[6], m_inH[cudaIn::ntime], 0x71374491);
        sharoundH(AH[6], AH[7], AH[0], AH[1], AH[2], AH[3], AH[4], AH[5], m_inH[cudaIn::nbits], 0xB5C0FBCF);
        sharoundLH(AH[5], AH[6], AH[7], AH[0], AH[1], AH[2], AH[3], AH[4], 0xE9B5DBA5);

        for (int i = 0; i < 8; ++i)
            m_inH[cudaIn::preCals + i] = AH[i];
*/
        CUDA_CHECK(cudaMemcpyAsync(m_inD, m_inH, cudaIn::count * sizeof (uint32), cudaMemcpyHostToDevice, stream));
    }

    int loops = GetStepIterations();
    int bits = GetStepBitShift() - 1;

    bitcoinSearch(stream, m_numb, m_numt, m_inD, m_outD, nonce, loops, bits);
    CUDA_CHECK(cudaMemcpyAsync(m_outH, m_outD, sizeof (uint32), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); //wait for a solution
    if (*m_outH != 0)
        return CryptoPP::ByteReverse(*m_outH);

    return 0;
}

/*
CUDARunner::CUDARunner():GPURunner<unsigned long,int>(TYPE_CUDA)
{
        m_in=0;
        m_devin=0;
        m_out=0;
        m_devout=0;

        cutilSafeCall(cudaGetDeviceCount(&m_devicecount));

        if(m_devicecount>0)
        {
                if(m_deviceindex<0 || m_deviceindex>=m_devicecount)
                {
                        m_deviceindex=cutGetMaxGflopsDeviceId();
                        printf("Setting CUDA device to Max GFlops device at index %u\n",m_deviceindex);
                }
                else
                {
                        printf("Setting CUDA device to device at index %u\n",m_deviceindex);
                }
		
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props,m_deviceindex);

                printf("Device info for %s :\nCompute Capability : %d.%d\nClock Rate (hz) : %d\n",props.name,props.major,props.minor,props.clockRate);

                if(props.major>999)
                {
                        printf("CUDA seems to be running in CPU emulation mode\n");
                }

                cutilSafeCall(cudaSetDevice(m_deviceindex));

        }
        else
        {
                m_deviceindex=-1;
                printf("No CUDA capable device detected\n");
        }
}

CUDARunner::~CUDARunner()
{
        DeallocateResources();
        cutilSafeCall(cudaThreadExit());
}

void CUDARunner::AllocateResources(const int numb, const int numt)
{
        DeallocateResources();

        m_in=(cuda_in *)malloc(sizeof(cuda_in));
        m_out=(cuda_out *)malloc(numb*numt*sizeof(cuda_out));

        cutilSafeCall(cudaMalloc((void **)&m_devin,sizeof(cuda_in)));
        cutilSafeCall(cudaMalloc((void **)&m_devout,numb*numt*sizeof(cuda_out)));

        printf("Done allocating CUDA resources for (%d,%d)\n",numb,numt);
}

void CUDARunner::DeallocateResources()
{
        if(m_in)
        {
                free(m_in);
                m_in=0;
        }
        if(m_devin)
        {
                cutilSafeCall(cudaFree(m_devin));
                m_devin=0;
        }
        if(m_out)
        {
                free(m_out);
                m_out=0;
        }
        if(m_devout)
        {
                cutilSafeCall(cudaFree(m_devout));
                m_devout=0;
        }
}

void CUDARunner::FindBestConfiguration()
{
        unsigned long lowb=16;
        unsigned long highb=128;
        unsigned long lowt=16;
        unsigned long hight=256;
        unsigned long bestb=16;
        unsigned long bestt=16;
        int64 besttime=std::numeric_limits<int64>::max();

        if(m_requestedgrid>0 && m_requestedgrid<=65536)
        {
                lowb=m_requestedgrid;
                highb=m_requestedgrid;
        }

        if(m_requestedthreads>0 && m_requestedthreads<=65536)
        {
                lowt=m_requestedthreads;
                hight=m_requestedthreads;
        }

        for(int numb=lowb; numb<=highb; numb*=2)
        {
                for(int numt=lowt; numt<=hight; numt*=2)
                {
                        AllocateResources(numb,numt);
                        // clear out any existing error
                        cudaError_t err=cudaGetLastError();
                        err=cudaSuccess;

                        int64 st=GetTimeMillis();

                        for(int it=0; it<128*256*2 && err==0; it+=(numb*numt))
                        {
                                cutilSafeCall(cudaMemcpy(m_devin,m_in,sizeof(cuda_in),cudaMemcpyHostToDevice));

                                cuda_process_helper(m_devin,m_devout,64,6,numb,numt);

                                cutilSafeCall(cudaMemcpy(m_out,m_devout,numb*numt*sizeof(cuda_out),cudaMemcpyDeviceToHost));

                                err=cudaGetLastError();
                                if(err!=cudaSuccess)
                                {
                                        printf("CUDA error %d\n",err);
                                }
                        }

                        int64 et=GetTimeMillis();

                        printf("Finding best configuration step end (%d,%d) %"PRI64d"ms  prev best=%"PRI64d"ms\n",numb,numt,et-st,besttime);

                        if((et-st)<besttime && err==cudaSuccess)
                        {
                                bestb=numb;
                                bestt=numt;
                                besttime=et-st;
                        }
                }
        }

        m_numb=bestb;
        m_numt=bestt;

        AllocateResources(m_numb,m_numt);

}

const unsigned long CUDARunner::RunStep()
{
        unsigned int best=0;
        unsigned int bestg=~0;

        if(m_in==0 || m_out==0 || m_devin==0 || m_devout==0)
        {
                AllocateResources(m_numb,m_numt);
        }

        cutilSafeCall(cudaMemcpy(m_devin,m_in,sizeof(cuda_in),cudaMemcpyHostToDevice));

        cuda_process_helper(m_devin,m_devout,GetStepIterations(),GetStepBitShift(),m_numb,m_numt);

        cutilSafeCall(cudaMemcpy(m_out,m_devout,m_numb*m_numt*sizeof(cuda_out),cudaMemcpyDeviceToHost));

        for(int i=0; i<m_numb*m_numt; i++)
        {
                if(m_out[i].m_bestnonce!=0 && m_out[i].m_bestg<bestg)
                {
                        best=m_out[i].m_bestnonce;
                        bestg=m_out[i].m_bestg;
                }
        }

        return CryptoPP::ByteReverse(best);

}
 */
