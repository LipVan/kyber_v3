/*
 * keccak.h
 *
 *  Created on: Dec 23, 2020
 *      Author: dcs
 */

#ifndef COMMON_KECCAK_H_
#define COMMON_KECCAK_H_

#include <stdint.h>
#include <stddef.h>


__device__ void KeccakF1600StatePermuteDev(uint64_t state[25]);

__device__ void KeccakAbsorbDev(uint64_t state[25],
                          unsigned int r,
                          const uint8_t *m,
                          size_t mlen,
                          uint8_t p);

__device__ void KeccakSqueezeblocksDev(uint8_t *out,
                                 size_t nblocks,
                                 uint64_t state[25],
                                 unsigned int r);


#endif /* COMMON_KECCAK_H_ */
