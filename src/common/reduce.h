/*
 * reduce.h
 *
 *  Created on: Dec 23, 2020
 *      Author: dcs
 */

#ifndef COMMON_REDUCE_H_
#define COMMON_REDUCE_H_

#include <stdint.h>
#include "params.h"

__device__ int16_t MontRdcDev(int32_t a);

__device__ int16_t BartRdcDev(int16_t a);

#endif /* COMMON_REDUCE_H_ */
