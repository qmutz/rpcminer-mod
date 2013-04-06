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

#if defined(_BITCOIN_MINER_CUDA_) || defined(_BITCOIN_MINER_OPENCL_)

#include "rpcminerthreadgpu.h"

RPCMinerThreadGPU::RPCMinerThreadGPU()
{
}

RPCMinerThreadGPU::~RPCMinerThreadGPU()
{
}

void RPCMinerThreadGPU::Run(void *arg)
{
    threaddata *td = (threaddata *) arg;
    int64 currentblockid = -1;

    static const unsigned int SHA256InitState[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint256 tempbuff[4];
    uint256 &temphash = *alignup < 16 > (tempbuff);
    uint256 hashbuff[4];
    uint256 &hash = *alignup < 16 > (hashbuff);
    uint256 currenttarget;

    unsigned char currentmidbuff[256];
    unsigned char currentblockbuff[256];
    volatile unsigned char *midbuffptr;
    volatile unsigned char *blockbuffptr;
    volatile unsigned int *nonce;
    unsigned int gpunonce;
    midbuffptr = alignup < 16 > (currentmidbuff);
    blockbuffptr = alignup < 16 > (currentblockbuff);
    nonce = (unsigned int *) (blockbuffptr + 12);

    gpurunnertype gpu;

    if (gpu.GetDeviceIndex() < 0)
    {
        return;
    }

    SetThreadPriority(THREAD_PRIORITY_LOWEST);

    FormatHashBlocks(&temphash, sizeof (temphash));
    for (int i = 0; i < 64 / 4; i++)
    {
        ((unsigned int*) &temphash)[i] = CryptoPP::ByteReverse(((unsigned int*) &temphash)[i]);
    }

    (*nonce) = 0;

    gpu.FindBestConfiguration();

    int64 threads = gpu.GetStepIterations() * gpu.GetNumBlocks();
    threads *= gpu.GetNumThreads();

    std::cout << "iterations on gpu=" << gpu.GetStepIterations() << " blocks=" << gpu.GetNumBlocks() << " threads=" << gpu.GetNumThreads() << std::endl;
    int64 iterations = int64(cudaIn::maxNonce) / (threads); //10000/(gpu.GetStepIterations()+gpu.GetNumBlocks()+gpu.GetNumThreads());
    printf("Iterations: %llu,hashs per iteration %llu\n", iterations, (gpu.GetStepIterations() * gpu.GetNumBlocks() * gpu.GetNumThreads()));
    unsigned int lastHash = 0u;
    unsigned int lastMerkle = 0u;
    unsigned int lstNTime = 0u;
    while (td->m_generate)
    {
        if (td->m_havework)
        {

            CRITICAL_BLOCK(td->m_cs);
            if (currentblockid != td->m_nextblock.m_blockid)
            {
                currenttarget = td->m_nextblock.m_target;
                currentblockid = td->m_nextblock.m_blockid;
                ::memcpy((void *) midbuffptr, &td->m_nextblock.m_midstate[0], 32);
                ::memcpy((void *) blockbuffptr, &td->m_nextblock.m_block[0], 64);
                ::memcpy(&temphash, &td->m_nextblock.m_hash1[0], 64);
                (*nonce) = 0;
                for (int i = 0; i < 8; i++)
                {
                    gpu.GetIn()[i] = ((unsigned int *) midbuffptr)[i];
                }
                lastMerkle = gpu.GetIn()[cudaIn::merkle];
                lstNTime = gpu.GetIn()[cudaIn::ntime];
                gpu.GetIn()[cudaIn::merkle] = ((unsigned int *) blockbuffptr)[0];
                gpu.GetIn()[cudaIn::ntime] = ((unsigned int *) blockbuffptr)[1];
                gpu.GetIn()[cudaIn::nbits] = ((unsigned int *) blockbuffptr)[2];
            }
            if (lastMerkle == gpu.GetIn()[cudaIn::merkle] && lstNTime == gpu.GetIn()[cudaIn::ntime])
            {
                printf("[ERROR] already nown solution\n");
                continue;
            }

            for (int64 i = 0; i < iterations; i++)
            {

                gpunonce = gpu.RunStep((*nonce));
                if (gpunonce != 0)
                {
                    (*nonce) = CryptoPP::ByteReverse(gpunonce);
                    printf("Found nonce %X\n",(*nonce));
                    SHA256Transform(&temphash, (void *) blockbuffptr, (void *) midbuffptr);
                    SHA256Transform(&hash, &temphash, SHA256InitState);

                    if (hash < currenttarget && lastHash != (*nonce))
                    {
                        CRITICAL_BLOCK(td->m_cs);

                        td->m_foundhashes.push_back(foundhash(currentblockid, (*nonce)));
                    }
                    lastHash = (*nonce);

                }

                (*nonce) += (gpu.GetStepIterations() * gpu.GetNumBlocks() * gpu.GetNumThreads());

                {
                    CRITICAL_BLOCK(td->m_cs);
                    td->m_hashcount += (gpu.GetStepIterations() * gpu.GetNumBlocks() * gpu.GetNumThreads());
                }

            }

        }
        else
        {
            Sleep(50);
        }

    }

    {
        CRITICAL_BLOCK(td->m_cs);
        td->m_done = true;
    }

}

#endif	// defined(_BITCOIN_MINER_CUDA_) || defined(_BITCOIN_MINER_OPENCL_)
