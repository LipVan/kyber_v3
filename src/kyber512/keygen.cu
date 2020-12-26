/*
 * keygen.cu
 *
 *  Created on: Dec 23, 2020
 *      Author: dcs
 */

#include "keygen.h"

__global__ void KeyGenKernel(uint8_t *sk, uint8_t *pk,  uint8_t *seed_d,
		int *s_bar, int *e_bar, int16_t *A_bar, int8_t *s, int8_t *e, int8_t *i_zeta_tab)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	uint8_t rho_sgm[2*KYBER_SEED_LEN];

	//rho,sigma = G(d)
	Sha3512Dev(rho_sgm, seed_d + tid*KYBER_SEED_LEN, KYBER_SEED_LEN);

	//Matrix A
	GenMtrxADev(A_bar + tid*KYBER_K*KYBER_K*KYBER_N, rho_sgm);

	//s ==> s_bar, e ==> e_bar
	{
		uint8_t nounce=0;
		GensreDev(s+tid*KYBER_K*KYBER_N, rho_sgm+KYBER_SEED_LEN, nounce);
		__syncthreads();
		RawNTTKernel_512(s_bar, s, i_zeta_tab);

		nounce += KYBER_K;
		GensreDev(e+tid*KYBER_K*KYBER_N, rho_sgm+KYBER_SEED_LEN, nounce);
		__syncthreads();
		RawNTTKernel_512(e_bar, e, i_zeta_tab);
	}

	//t_bar = A_bar * t_bar + e_bar


}


