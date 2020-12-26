/*
 * ntt.h
 *
 *  Created on: Dec 25, 2020
 *      Author: dcs
 */

#ifndef COMMON_NTT_H_
#define COMMON_NTT_H_

#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#define SHARED_MEM_SIZE		(64*KYBER_N*2)
#define COPY_STRIDE			(BLOCK_SIZE*sizeof(int4))
#define COPY_ROUNDS			(SHARED_MEM_SIZE / COPY_STRIDE)

#define TILES_PER_MTRX_ROW	(KYBER_N / WMMA_N)
#define TILES_PER_WARP_ROW	(TILES_PER_MTRX_ROW / WARPS_PER_BLOCK)

#define WMMA_ROW		1
#define WMMA_ROUNDS		((KYBER_N / WMMA_K) / WMMA_ROW)

#define WMMA_CONCURRENCE	2

#ifdef KYBER_512
#define ITERATIONS	(BLOCK_SIZE*2*KYBER_N / SHARED_MEM_SIZE)
#define ROUNDS_IN_ITERATION	(SHARED_MEM_SIZE / (KYBER_N * 2 * WMMA_M))
#endif

__global__ void RawNTTKernel_512(int *o_sre, int8_t *i_sre, int8_t *i_zeta_tb);

__device__ void PointWiseAccDev(int16_t *t_bar, int16_t *A_bar, int *s_bar, int *e_bar);

#endif /* COMMON_NTT_H_ */
