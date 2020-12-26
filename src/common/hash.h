/*
 * hash.h
 *
 *  Created on: Dec 23, 2020
 *      Author: dcs
 */

#ifndef COMMON_HASH_H_
#define COMMON_HASH_H_

#include <stdint.h>
#include <stddef.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "keccak.h"

#define SHAKE_SUFFIX 0x1F
#define SHA3_SUFFIX 0x06

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#define SHA3_256_RATE 136
#define SHA3_512_RATE 72

__device__ void Shake128Dev(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen);

__device__ void Shake256Dev(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen);

__device__ void Sha3256Dev(uint8_t h[32], const uint8_t *in, size_t inlen);

__device__ void Sha3512Dev(uint8_t *h, const uint8_t *in, size_t inlen);


#endif /* COMMON_HASH_H_ */
