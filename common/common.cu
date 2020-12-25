/*
 * common.cu
 *
 *  Created on: Dec 23, 2020
 *      Author: dcs
 */

#include "common.h"

__device__ void CBDEtaDev(int8_t *o_poly, uint8_t *i_B)
{
	for(int i=0; i<KYBER_N/8; ++i)
	{
		uint32_t t = (uint32_t)*(i_B + 4*i);
#pragma unroll
		for(int pos=1; pos<4; ++pos)
		{
			t |= (uint32_t)*(i_B + 4*i + pos) << (pos*8);
		}

		uint32_t d = t & 0x55555555;
		d += (t >> 1) & 0x55555555;

#pragma unroll
		for(int j=0; j<8; ++j)
		{
			int8_t a = (d >> (4*j+0)) & 0x3;
			int8_t b = (d >> (4*j+2)) & 0x3;

			o_poly[8*i+j] = a-b;
		}
	}
}


__device__ void CBDEtaDevLoc(int8_t *o_poly, uint8_t *i_B)
{
	uint8_t tmp;
	int8_t a;
	int8_t b;

#pragma unroll
	for(int i=0; i<64*KYBER_ETA; ++i)
	{
#pragma unroll
		for(int j=0; j<2; ++j)
		{
			tmp = (i_B[i] >> (1-j)*4) &0xf;
			a = (tmp & 0x8) >> 3 + (tmp & 0x4)>>2;
			b = (tmp & 0x2) >> 1 + (tmp & 0x1);
			o_poly[i*2 + j] = a-b;
		}
	}
}

__device__ int ParseDev(int16_t *o_poly, uint req_len, uint8_t *byts, uint byts_len)
{
	uint ctr=0, pos=0;
	uint16_t d1=0, d2=0;

	while(ctr<req_len && pos+3 < byts_len)
	{
		d1 = ((byts[pos+0] >> 0) | ((uint16_t)byts[pos+1] << 8)) & 0xFFF;
		d2 = ((byts[pos+2] >> 4) | ((uint16_t)byts[pos+2] << 4)) & 0xFFF;
		pos += 3;

		if(d1 < KYBER_Q)
			o_poly[ctr++] = d1;
		if(ctr<req_len && d2<KYBER_Q)
			o_poly[ctr++] = d2;
	}

	return ctr;
}

#define XOF_BLOCKBYTES SHAKE128_RATE
#define GEN_MATRIX_NBLOCKS ((12*KYBER_N/8*(1 << 12)/KYBER_Q + XOF_BLOCKBYTES)/XOF_BLOCKBYTES)

__device__ void GenMtrxADev(int16_t *mtrx_A, uint8_t *rho)
{
	uint64_t state[25];
	uint8_t extseed[KYBER_SEED_LEN+2];
	uint8_t buf[GEN_MATRIX_NBLOCKS*XOF_BLOCKBYTES+2];

	for(int i=0; i<KYBER_SEED_LEN; ++i)
		extseed[i] = rho[i];

	for(int i=0; i<KYBER_K; ++i)
	{
		for(int j=0; j<KYBER_K; ++j)
		{
			extseed[KYBER_SEED_LEN] = j;
			extseed[KYBER_SEED_LEN+1] = i;

			KeccakAbsorbDev(state, SHAKE128_RATE, extseed, KYBER_SEED_LEN+2, 0x1F);
			KeccakSqueezeblocksDev(buf, GEN_MATRIX_NBLOCKS, state, SHAKE128_RATE);

			uint buf_len = GEN_MATRIX_NBLOCKS*XOF_BLOCKBYTES;
			uint ctr = ParseDev(mtrx_A + i*KYBER_K*KYBER_N + j*KYBER_N, KYBER_N, buf, buf_len);
		    while(ctr < KYBER_N)
		    {
		      uint off = buf_len % 3;
		      for(int k = 0; k < off; k++)
		        buf[k] = buf[buf_len - off + k];
		      KeccakSqueezeblocksDev(buf+off, 1, state, SHAKE128_RATE);
		      buf_len = off + XOF_BLOCKBYTES;

		      ctr += ParseDev(mtrx_A+ctr, KYBER_N-ctr, buf, buf_len);
		    }
		}
	}
}


__device__ void GensreDev(int8_t *o_sre, uint8_t *seed, uint8_t nounce)
{
	uint8_t buf[KYBER_ETA*KYBER_N / 4];
	uint 	out_len = KYBER_ETA*KYBER_N / 4;
	uint8_t extseed[KYBER_SEED_LEN+1];

	for(int i=0; i<KYBER_SEED_LEN; ++i)
		extseed[i] = seed[i];

	for(int i=0; i<KYBER_K; ++i)
	{
		uint64_t state[25];
		uint8_t tmp_buf[SHAKE256_RATE];
		extseed[KYBER_SEED_LEN] = nounce+i;

		KeccakAbsorbDev(state, SHAKE256_RATE, extseed, KYBER_SEED_LEN+1, SHAKE_SUFFIX);
		KeccakSqueezeblocksDev((unsigned char *)buf, out_len/ SHAKE256_RATE, state, SHAKE256_RATE);

		out_len -= (out_len/SHAKE256_RATE) * SHAKE256_RATE;
		if(out_len)
		{
			KeccakSqueezeblocksDev(tmp_buf, 1, state, SHAKE256_RATE);
			for(int j=0; j<out_len; ++j)
				buf[((KYBER_ETA*KYBER_N/4)/SHAKE256_RATE)*SHAKE256_RATE + j] = tmp_buf[j];
		}

		CBDEtaDev(o_sre+i*KYBER_N, buf);
	}
}
