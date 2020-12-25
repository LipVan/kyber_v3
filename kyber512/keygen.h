/*
 * keygen.h
 *
 *  Created on: Dec 23, 2020
 *      Author: dcs
 */

#ifndef KYBER512_KEYGEN_H_
#define KYBER512_KEYGEN_H_

#include "../common/common.h"

__global__ void KeyGenKernel(uint8_t *sk, uint8_t *pk, uint8_t *seed_d);


#endif /* KYBER512_KEYGEN_H_ */
