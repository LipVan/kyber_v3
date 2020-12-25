/*
 * reduce.cu
 *
 *  Created on: Dec 23, 2020
 *      Author: dcs
 */
#include "reduce.h"

/*************************************************
* Name:        montgomery_reduce
*
* Description: Montgomery reduction; given a 32-bit integer a, computes
*              16-bit integer congruent to a * R^-1 mod q,
*              where R=2^16
*
* Arguments:   - int32_t a: input integer to be reduced;
*                           has to be in {-q2^15,...,q2^15-1}
*
* Returns:     integer in {-q+1,...,q-1} congruent to a * R^-1 modulo q.
**************************************************/
__device__ int16_t MontRdcDev(int32_t a)
{
  int32_t t;
  int16_t u;

  u = a*KYBER_QINV;
  t = (int32_t)u*KYBER_Q;
  t = a - t;
  t >>= 16;
  return t;
}

/*************************************************
* Name:        barrett_reduce
*
* Description: Barrett reduction; given a 16-bit integer a, computes
*              16-bit integer congruent to a mod q in {0,...,q}
*
* Arguments:   - int16_t a: input integer to be reduced
*
* Returns:     integer in {0,...,q} congruent to a modulo q.
**************************************************/
__device__ int16_t BartRdcDev(int16_t a)
{
	  int16_t t;
	  const int16_t v = ((1U << 26) + KYBER_Q/2)/KYBER_Q;

	  t  = (int32_t)v*a >> 26;
	  t *= KYBER_Q;
	  return a - t;
}
