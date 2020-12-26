/*
 * ntt.cu
 *
 *  Created on: Dec 25, 2020
 *      Author: dcs
 */
#include "ntt.h"

using namespace nvcuda;

//Using 32kB shared memory
__global__ void RawNTTKernel_512(int *o_sre, int8_t *i_sre, int8_t *i_zeta_tb)
{
	//32 kB sre = 16 * ( 256*2 ) * 4
	__shared__ int8_t shmem[SHARED_MEM_SIZE];

	int tid = threadIdx.x;
	const int warp_id = tid / WARP_SIZE;
	int lane_id = tid % WARP_SIZE;


	for(int ite=0; ite<ITERATIONS; ++ite)
	{
#pragma unroll
		//Copy sre from global to shared memory
		for(int cp_ind=0; cp_ind<COPY_ROUNDS; ++cp_ind)
		{
			*((int4 *)(shmem + cp_ind*COPY_STRIDE) + tid) = \
					*((int4 *)(i_sre + blockIdx.x*BLOCK_SIZE*2*KYBER_N + ite*SHARED_MEM_SIZE + cp_ind*COPY_STRIDE) + lane_id);
		}

		wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, char, wmma::row_major> a_frag;		//WMMA_CONCURRENCE
		wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, char, wmma::row_major> b_frag[TILES_PER_WARP_ROW]; // 1 row; 2 row?
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, int> c_frag[WMMA_CONCURRENCE][TILES_PER_WARP_ROW];

		//init c_frag to 0
		for(int i=0; i<WMMA_CONCURRENCE; ++i)
		{
			for(int j=0; j<TILES_PER_WARP_ROW; ++j)
			{
				wmma::fill_fragment(c_frag[i][j], 0);
			}
		}
		__syncthreads();

		//Iterate on shared memory
		for(int ite_on_sh=0; ite_on_sh<ROUNDS_IN_ITERATION; ++ite_on_sh)
		{

			//c = a * b_h, the high value of zeta table
			for(int round=0; round<WMMA_ROUNDS; ++round)
			{
				//Load zeta tables from global memory to fragment
#pragma unroll
				for(int i=0; i<TILES_PER_WARP_ROW; ++i)
				{
					char *zeta_warp_ptr = (char *)(i_zeta_tb + warp_round*KYBER_N*WMMA_K + warp_id*TILES_PER_WARP_ROW*WMMA_N);

					wmma::load_matrix_sync(b_frag[i], (char *)zeta_warp_ptr+i*WMMA_K, KYBER_N);
				}

				//Load sre from shared memory to fragment and perform mma
#pragma unroll
				for(int i=0; i<WMMA_CONCURRENCE; ++i)
				{
					char *sre_warp_ptr = (char *)(shmem + ite_on_sh*KYBER_N*2*WMMA_M);

					wmma::load_matrix_sync(a_frag[i], sre_warp_ptr + round*WMMA_K + i*KYBER_N*WMMA_M, KYBER_N);

#pragma unroll
					for(int j=0; j<TILES_PER_WARP_ROW; ++j)
					{
						wmma::mma_sync(c_frag[i][j], a_frag, b_frag[j], c_frag[i][j]);
					}
				}
			}

			//c = c<< 7
#pragma unroll
			for(int i=0; i<WMMA_CONCURRENCE; ++i)
			{
				for(int j=0; j<TILES_PER_WARP_ROW; ++j)
				{
					for(int t=0; t<c_frag[i][j].num_elements; ++t)
					{
						c_frag[i][j].x[t] *= BASE_BITS;
						//c_frag[i][j].x[t] <<= 7;
					}
				}
			}

			//c = c + a*b_l , the low value of zeta table
			for(int round=0; round<WMMA_ROUNDS; ++round)
			{
				//Load zeta tables from global memory to fragment
#pragma unroll
				for(int i=0; i<TILES_PER_WARP_ROW; ++i)
				{
					char *zeta_warp_ptr = (char *)(i_zeta_tb + KYBER_N*KYBER_N + \
							warp_round*KYBER_N*WMMA_K + warp_id*TILES_PER_WARP_ROW*WMMA_N);

					wmma::load_matrix_sync(b_frag[i], (char *)zeta_warp_ptr + i*WMMA_K, KYBER_N);
				}

				//Load sre from shared memory to fragment and perform mma
#pragma unroll
				for(int i=0; i<WMMA_CONCURRENCE; ++i)
				{
					char *sre_warp_ptr = (char *)(shmem + ite_on_sh*KYBER_N*2*WMMA_M);

					wmma::load_matrix_sync(a_frag[i], sre_warp_ptr + round*WMMA_K + i*KYBER_N*WMMA_M, KYBER_N);

#pragma unroll
					for(int j=0; j<TILES_PER_WARP_ROW; ++j)
					{
						wmma::mma_sync(c_frag[i][j], a_frag, b_frag[j], c_frag[i][j]);
					}
				}
			}
		}

		//store the result
		for(int i=0; i<WMMA_CONCURRENCE; ++i)
		{
			for(int j=0; j<TILES_PER_WARP_ROW; ++j)
			{
				int *o_sre_warp_ptr = (int *)o_sre + ite*SHARED_MEM_SIZE/sizeof(char) + warp_id*TILES_PER_WARP_ROW*WMMA_N;

				wmma::store_matrix_sync((o_sre_warp_ptr + j*WMMA_N + i*KYBER_N*WMMA_M), c_frag[i][j], KYBER_N, wmma::mem_row_major);
			}
		}
	}
}


__device__ void PointWiseAccDev(int16_t *t_bar, int16_t *A_bar, int *s_bar, int *e_bar)
{
	//Based on vec t_bar
	//12-bit + (12 + 8 + 2)bit = 34 bit -> 16 bit
	for(int col=0; col<KYBER_K; ++col)
	{
		int64_t tmp_v = 0;
		for(int ind=0; ind<KYBER_N; ++ind)
		{

		}
	}

}
