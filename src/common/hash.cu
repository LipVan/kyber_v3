/*
 * hash.cu
 *
 *  Created on: Dec 23, 2020
 *      Author: dcs
 */
#include "hash.h"

__device__ void Shake128Dev(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen)
{
  unsigned int i;
  size_t nblocks = outlen/SHAKE128_RATE;
  uint8_t t[SHAKE128_RATE];
  uint64_t state[25];

  KeccakAbsorbDev(state, SHAKE128_RATE, in, inlen, 0x1F);
  KeccakSqueezeblocksDev(out, nblocks, state, SHAKE128_RATE);

  out += nblocks*SHAKE128_RATE;
  outlen -= nblocks*SHAKE128_RATE;

  if(outlen) {
    KeccakSqueezeblocksDev(t, 1, state, SHAKE128_RATE);
    for(i=0;i<outlen;i++)
      out[i] = t[i];
  }
}

/*************************************************
* Name:        shake256
*
* Description: SHAKE256 XOF with non-incremental API
*
* Arguments:   - uint8_t *out:      pointer to output
*              - size_t outlen:     requested output length in bytes
*              - const uint8_t *in: pointer to input
*              - size_t inlen:      length of input in bytes
**************************************************/
__device__ void Shake256Dev(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen)
{
  unsigned int i;
  size_t nblocks = outlen/SHAKE256_RATE;
  uint8_t t[SHAKE256_RATE];
  uint64_t state[25];

  KeccakAbsorbDev(state, SHAKE256_RATE, in, inlen, SHAKE_SUFFIX);
  KeccakSqueezeblocksDev(out, nblocks, state, SHAKE256_RATE);

  out += nblocks*SHAKE256_RATE;
  outlen -= nblocks*SHAKE256_RATE;

  if(outlen) {
    KeccakSqueezeblocksDev(t, 1, state, SHAKE256_RATE);
    for(i=0;i<outlen;i++)
      out[i] = t[i];
  }
}

/*************************************************
* Name:        sha3_256
*
* Description: SHA3-256 with non-incremental API
*
* Arguments:   - uint8_t *h:        pointer to output (32 bytes)
*              - const uint8_t *in: pointer to input
*              - size_t inlen:      length of input in bytes
**************************************************/
__device__ void Sha3256Dev(uint8_t h[32], const uint8_t *in, size_t inlen)
{
  unsigned int i;
  uint64_t state[25];
  uint8_t t[SHA3_256_RATE];

  KeccakAbsorbDev(state, SHA3_256_RATE, in, inlen, 0x06);
  KeccakSqueezeblocksDev(t, 1, state, SHA3_256_RATE);

  for(i=0;i<32;i++)
    h[i] = t[i];
}

/*************************************************
* Name:        sha3_512
*
* Description: SHA3-512 with non-incremental API
*
* Arguments:   - uint8_t *h:        pointer to output (64 bytes)
*              - const uint8_t *in: pointer to input
*              - size_t inlen:      length of input in bytes
**************************************************/
__device__ void Sha3512Dev(uint8_t *h, const uint8_t *in, size_t inlen)
{
  unsigned int i;
  uint64_t state[25];
  uint8_t t[SHA3_512_RATE];

  KeccakAbsorbDev(state, SHA3_512_RATE, in, inlen, 0x06);
  KeccakSqueezeblocksDev(t, 1, state, SHA3_512_RATE);

  for(i=0;i<64;i++)
    h[i] = t[i];
}


