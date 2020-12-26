/*
 * params.h
 *
 *  Created on: Dec 23, 2020
 *      Author: dcs
 */

#ifndef COMMON_PARAMS_H_
#define COMMON_PARAMS_H_

#define KYBER_N 256
#define KYBER_Q 3329
#define KYBER_SEED_LEN	32

#define KYBER_MONT 2285 // 2^16 mod q
#define KYBER_QINV 62209 // q^-1 mod 2^16

#define KYBER_512

#ifdef KYBER_512
#define KYBER_K		2
#define KYBER_ETA	2
#define KYBER_DU	10
#define KYBER_DV	4
#endif

#ifdef KYBER_768
#define KYBER_K		3
#define KYBER_ETA	2
#define KYBER_DU	10
#define KYBER_DV	4
#endif

#ifdef KYBER_1024
#define KYBER_K		4
#define KYBER_ETA	2
#define KYBER_DU	11
#define KYBER_DV	5
#endif

#endif /* COMMON_PARAMS_H_ */
