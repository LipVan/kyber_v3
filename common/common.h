/*
 * common.h
 *
 *  Created on: Dec 23, 2020
 *      Author: dcs
 */

#ifndef COMMON_COMMON_H_
#define COMMON_COMMON_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "params.h"
#include "keccak.h"
#include "hash.h"
#include "reduce.h"

extern "C"{
#include "rng.h"
}

#define GRID_SIZE	(68)
#define BLOCK_SIZE	(128)

#define WARP_SIZE	32
#define WARPS_PER_BLOCK	(BLOCK_SIZE / WARP_SIZE)

#define WMMA_M	16
#define WMMA_N	16
#define WMMA_K	16

#define BASE_BITS	(2^7)

__device__ void CBDEtaDev(int8_t *o_poly, uint8_t i_B);

__device__ void CBDEtaDevLoc(int8_t *o_poly, uint8_t *i_B);

__device__ int ParseDev(int16_t *o_poly, uint req_len, uint8_t *byts, uint byts_len);

__device__ void GenMtrxADev(int16_t *mtrx_A, uint8_t *rho);

__device__ void GensreDev(int8_t *o_sre, uint8_t *seed, uint8_t nounce);

#endif /* COMMON_COMMON_H_ */
