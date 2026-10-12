/***********************************************************************
silk_port.c -- the analysis parts of the SILK speech codec, ported from
libopus (https://opus-codec.org, git 503d81b1) for rt_lpc:

  - LPC estimation by Burg's method      (silk/float/burg_modified_FLP.c)
  - the pitch estimator                  (silk/float/find_pitch_lags_FLP.c,
                                          silk/float/pitch_analysis_core_FLP.c)
  - the voice activity detector          (silk/VAD.c)
  and what they use: autocorrelation, Schur, k2a, bandwidth expansion,
  sine window, LPC analysis filter, sorting, the 2x and 3/2 downsamplers,
  the analysis filter bank, lin2log, sigmoid, the pitch codebook tables,
  and CELT's pitch cross-correlation.

The code is libopus's float build, copied as is, except:
  - libopus's headers are replaced by the definitions below (types,
    fixed-point macros, constants), copied from those headers
  - SILK_MAX_ORDER_LPC is raised from 24 to 128 (rt_lpc allows orders up
    to 100; the algorithms don't depend on it, only array sizes do)
  - assertions are off, as in a release build of libopus
  - CELT's pitch cross-correlation is its plain C version (same sums)
  - silk_find_pitch_lags_FLP() and the VAD's encoder-state argument are
    adapted to standalone state (see the end of this file), and
    silk_port_*() functions are added as the interface (silk_port.h)
  - one departure from SILK's behavior, for voice transformation rather
    than coding, which can be turned off to get SILK's exactly: the pitch
    search also runs in frames the VAD finds no voice activity in
    (silk_port_pitch_set_vad_gating())

Original copyright and license:

Copyright (c) 2006-2011, Skype Limited. All rights reserved.
Copyright (c) 2007-2008 CSIRO, Copyright (c) 2007-2009 Xiph.Org Foundation
(celt_pitch_xcorr)
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <alloca.h>
#include "silk_port.h"

/*=====================================================================
  Definitions from libopus's headers
  =====================================================================*/

/* include/opus_types.h, celt/arch.h (float build) */
typedef signed char        opus_int8;
typedef unsigned char      opus_uint8;
typedef short              opus_int16;
typedef unsigned short     opus_uint16;
typedef int                opus_int32;
typedef unsigned int       opus_uint32;
typedef long long          opus_int64;
typedef unsigned long long opus_uint64;
typedef int                opus_int;
typedef unsigned int       opus_uint;
typedef float              silk_float;
typedef float              opus_val16;
typedef float              opus_val32;
#define OPUS_INLINE        inline
#define MAC16_16(c,a,b)    ((c)+(opus_val32)(a)*(opus_val32)(b))

/* assertions: off, as in a release build */
#define silk_assert(x)     ((void)0)
#define celt_assert(x)     ((void)0)
#define celt_sig_assert(x) ((void)0)

/* celt/stack_alloc.h (alloca variant) */
#define VARDECL(type, var) type *var
#define ALLOC(var, size, type) var = ((type*)alloca(sizeof(type)*(size)))
#define SAVE_STACK
#define RESTORE_STACK

/* celt/float_cast.h */
#define float2int(x)       lrintf(x)

/* silk/typedef.h */
#define silk_int64_MAX     ((opus_int64)0x7FFFFFFFFFFFFFFFLL)
#define silk_int64_MIN     ((opus_int64)0x8000000000000000LL)
#define silk_int32_MAX     0x7FFFFFFF
#define silk_int32_MIN     ((opus_int32)0x80000000)
#define silk_int16_MAX     0x7FFF
#define silk_int16_MIN     ((opus_int16)0x8000)
#define silk_int8_MAX      0x7F
#define silk_int8_MIN      ((opus_int8)0x80)
#define silk_uint8_MAX     0xFF

/* silk/macros.h */
#define silk_SMULWB(a32, b32)            ((opus_int32)(((a32) * (opus_int64)((opus_int16)(b32))) >> 16))
#define silk_SMLAWB(a32, b32, c32)       ((opus_int32)((a32) + (((b32) * (opus_int64)((opus_int16)(c32))) >> 16)))
#define silk_SMULWT(a32, b32)            ((opus_int32)(((a32) * (opus_int64)((b32) >> 16)) >> 16))
#define silk_SMLAWT(a32, b32, c32)       ((opus_int32)((a32) + (((b32) * ((opus_int64)(c32) >> 16)) >> 16)))
#define silk_SMULBB(a32, b32)            ((opus_int32)((opus_int16)(a32)) * (opus_int32)((opus_int16)(b32)))
#define silk_SMLABB(a32, b32, c32)       ((a32) + ((opus_int32)((opus_int16)(b32))) * (opus_int32)((opus_int16)(c32)))
#define silk_SMULBT(a32, b32)            ((opus_int32)((opus_int16)(a32)) * ((b32) >> 16))
#define silk_SMLABT(a32, b32, c32)       ((a32) + ((opus_int32)((opus_int16)(b32))) * ((c32) >> 16))
#define silk_SMLAL(a64, b32, c32)        (silk_ADD64((a64), ((opus_int64)(b32) * (opus_int64)(c32))))
#define silk_SMULWW(a32, b32)            ((opus_int32)(((opus_int64)(a32) * (b32)) >> 16))
#define silk_SMLAWW(a32, b32, c32)       ((opus_int32)((a32) + (((opus_int64)(b32) * (c32)) >> 16)))
#define silk_ADD_SAT32(a, b)             ((((opus_uint32)(a) + (opus_uint32)(b)) & 0x80000000) == 0 ?                              \
                                        ((((a) & (b)) & 0x80000000) != 0 ? silk_int32_MIN : (a)+(b)) :   \
                                        ((((a) | (b)) & 0x80000000) == 0 ? silk_int32_MAX : (a)+(b)) )
#define silk_SUB_SAT32(a, b)             ((((opus_uint32)(a)-(opus_uint32)(b)) & 0x80000000) == 0 ?                                        \
                                        (( (a) & ((b)^0x80000000) & 0x80000000) ? silk_int32_MIN : (a)-(b)) :    \
                                        ((((a)^0x80000000) & (b)  & 0x80000000) ? silk_int32_MAX : (a)-(b)) )
#define matrix_ptr(Matrix_base_adr, row, column, N) \
    (*((Matrix_base_adr) + ((row)*(N)+(column))))
static OPUS_INLINE opus_int32 silk_CLZ32( opus_int32 in32 )
{
    return in32 ? __builtin_clz( (opus_uint32)in32 ) : 32;
}

/* silk/SigProc_FIX.h */
#define SILK_MAX_ORDER_LPC            128   /* libopus: 24 */
#define silk_MUL(a32, b32)                  ((a32) * (b32))
#define silk_MLA(a32, b32, c32)             silk_ADD32((a32),((b32) * (c32)))
#define silk_ADD32(a, b)                    ((a) + (b))
#define silk_ADD64(a, b)                    ((a) + (b))
#define silk_SUB32(a, b)                    ((a) - (b))
#define silk_SAT16(a)                       ((a) > silk_int16_MAX ? silk_int16_MAX :      \
                                            ((a) < silk_int16_MIN ? silk_int16_MIN : (a)))
#define silk_SAT32(a)                       ((a) > silk_int32_MAX ? silk_int32_MAX :      \
                                            ((a) < silk_int32_MIN ? silk_int32_MIN : (a)))
#define silk_ADD_SAT16(a, b)                (opus_int16)silk_SAT16( silk_ADD32( (opus_int32)(a), (b) ) )
#define silk_ADD_POS_SAT32(a, b)            ((((opus_uint32)(a)+(opus_uint32)(b)) & 0x80000000) ? silk_int32_MAX : ((a)+(b)))
#define silk_LSHIFT32(a, shift)             ((opus_int32)((opus_uint32)(a)<<(shift)))
#define silk_LSHIFT(a, shift)               silk_LSHIFT32(a, shift)
#define silk_RSHIFT32(a, shift)             ((a)>>(shift))
#define silk_RSHIFT(a, shift)               silk_RSHIFT32(a, shift)
#define silk_ADD_LSHIFT32(a, b, shift)      silk_ADD32((a), silk_LSHIFT32((b), (shift)))
#define silk_ADD_RSHIFT32(a, b, shift)      silk_ADD32((a), silk_RSHIFT32((b), (shift)))
#define silk_RSHIFT_ROUND(a, shift)         ((shift) == 1 ? ((a) >> 1) + ((a) & 1) : (((a) >> ((shift) - 1)) + 1) >> 1)
#define silk_DIV32_16(a32, b16)             ((opus_int32)((a32) / (b16)))
#define silk_DIV32(a32, b32)                ((opus_int32)((a32) / (b32)))
#define silk_min(a, b)                      (((a) < (b)) ? (a) : (b))
#define silk_max(a, b)                      (((a) > (b)) ? (a) : (b))
#define SILK_FIX_CONST( C, Q )              ((opus_int32)((C) * ((opus_int64)1 << (Q)) + 0.5))
#define silk_LIMIT( a, limit1, limit2)      ((limit1) > (limit2) ? ((a) > (limit1) ? (limit1) : ((a) < (limit2) ? (limit2) : (a))) \
                                                                 : ((a) > (limit2) ? (limit2) : ((a) < (limit1) ? (limit1) : (a))))
#define silk_LIMIT_int                      silk_LIMIT
#define silk_LIMIT_32                       silk_LIMIT
#define silk_memset(dest, src, size)        memset((dest), (src), (size))
#define silk_memcpy(dest, src, size)        memcpy((dest), (src), (size))
#define silk_memmove(dest, src, size)       memmove((dest), (src), (size))
#define silk_DWORD_ALIGN
static OPUS_INLINE opus_int silk_min_int(opus_int a, opus_int b) { return (((a) < (b)) ? (a) : (b)); }
static OPUS_INLINE opus_int silk_max_int(opus_int a, opus_int b) { return (((a) > (b)) ? (a) : (b)); }
static OPUS_INLINE opus_int32 silk_max_32(opus_int32 a, opus_int32 b) { return (((a) > (b)) ? (a) : (b)); }
static OPUS_INLINE opus_int32 silk_ROR32( opus_int32 a32, opus_int rot )
{
    opus_uint32 x = (opus_uint32) a32;
    opus_uint32 r = (opus_uint32) rot;
    opus_uint32 m = (opus_uint32) -rot;
    if( rot == 0 ) {
        return a32;
    } else if( rot < 0 ) {
        return (opus_int32) ((x << m) | (x >> (32 - m)));
    } else {
        return (opus_int32) ((x << (32 - r)) | (x >> r));
    }
}
static opus_int silk_sigm_Q15( opus_int in_Q5 );
static opus_int32 silk_lin2log( const opus_int32 inLin );

/* silk/Inlines.h */
static OPUS_INLINE void silk_CLZ_FRAC( opus_int32 in, opus_int32 *lz, opus_int32 *frac_Q7 )
{
    opus_int32 lzeros = silk_CLZ32(in);
    * lz = lzeros;
    * frac_Q7 = silk_ROR32(in, 24 - lzeros) & 0x7f;
}
static OPUS_INLINE opus_int32 silk_SQRT_APPROX( opus_int32 x )
{
    opus_int32 y, lz, frac_Q7;
    if( x <= 0 ) {
        return 0;
    }
    silk_CLZ_FRAC(x, &lz, &frac_Q7);
    if( lz & 1 ) {
        y = 32768;
    } else {
        y = 46214;        /* 46214 = sqrt(2) * 32768 */
    }
    /* get scaling right */
    y >>= silk_RSHIFT(lz, 1);
    /* increment using fractional part of input */
    y = silk_SMLAWB(y, y, silk_SMULBB(213, frac_Q7));
    return y;
}

/* silk/float/SigProc_FLP.h */
#define PI               (3.1415926536f)
#define silk_max_float( a, b )                  (((a) > (b)) ? (a) :  (b))
#define silk_abs_float( a )                     ((silk_float)fabs(a))
#define silk_inner_product_FLP(data1, data2, dataSize, arch) ((void)arch,silk_inner_product_FLP_c(data1, data2, dataSize))
#define silk_autocorrelation_FLP_arch           silk_autocorrelation_FLP
static double silk_inner_product_FLP_c( const silk_float *data1, const silk_float *data2, opus_int dataSize );
static double silk_energy_FLP( const silk_float *data, opus_int dataSize );
static OPUS_INLINE void silk_float2short_array( opus_int16 *out, const silk_float *in, opus_int32 length )
{
    opus_int32 k;
    for( k = length - 1; k >= 0; k-- ) {
        out[k] = silk_SAT16( (opus_int32)float2int( in[k] ) );
    }
}
static OPUS_INLINE void silk_short2float_array( silk_float *out, const opus_int16 *in, opus_int32 length )
{
    opus_int32 k;
    for( k = length - 1; k >= 0; k-- ) {
        out[k] = (silk_float)in[k];
    }
}
static OPUS_INLINE silk_float silk_log2( double x )
{
    return ( silk_float )( 3.32192809488736 * log10( x ) );
}
static void silk_insertion_sort_decreasing_FLP( silk_float *a, opus_int *idx, const opus_int L, const opus_int K );

/* silk/define.h */
#define MAX_FS_KHZ                              16
#define MAX_NB_SUBFR                            4
#define SUB_FRAME_LENGTH_MS                     5
#define MAX_FRAME_LENGTH_MS                     ( SUB_FRAME_LENGTH_MS * MAX_NB_SUBFR )
#define MAX_FRAME_LENGTH                        ( MAX_FRAME_LENGTH_MS * MAX_FS_KHZ )
#define LTP_MEM_LENGTH_MS                       20
#define LA_PITCH_MS                             2
#define MAX_FIND_PITCH_LPC_ORDER                16
#define FIND_PITCH_LPC_WIN_MS                   ( 20 + (LA_PITCH_MS << 1) )
#define FIND_PITCH_LPC_WIN_MAX                  ( FIND_PITCH_LPC_WIN_MS * MAX_FS_KHZ )
#define TYPE_NO_VOICE_ACTIVITY                  0
#define TYPE_UNVOICED                           1
#define TYPE_VOICED                             2
#define VAD_N_BANDS                             4
#define VAD_INTERNAL_SUBFRAMES_LOG2             2
#define VAD_INTERNAL_SUBFRAMES                  ( 1 << VAD_INTERNAL_SUBFRAMES_LOG2 )
#define VAD_NOISE_LEVEL_SMOOTH_COEF_Q16         1024
#define VAD_NOISE_LEVELS_BIAS                   50
#define VAD_NEGATIVE_OFFSET_Q5                  128
#define VAD_SNR_FACTOR_Q16                      45000
#define VAD_SNR_SMOOTH_COEF_Q18                 4096

/* silk/tuning_parameters.h */
#define FIND_PITCH_WHITE_NOISE_FRACTION         1e-3f
#define FIND_PITCH_BANDWIDTH_EXPANSION          0.99f
#define FIND_LPC_COND_FAC                       1e-5f
#define SPEECH_ACTIVITY_DTX_THRES               0.05f

/* silk/pitch_est_defines.h */
#define PE_MAX_FS_KHZ               16
#define PE_MAX_NB_SUBFR             4
#define PE_SUBFR_LENGTH_MS          5
#define PE_LTP_MEM_LENGTH_MS        ( 4 * PE_SUBFR_LENGTH_MS )
#define PE_MAX_FRAME_LENGTH_MS      ( PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS )
#define PE_MAX_FRAME_LENGTH         ( PE_MAX_FRAME_LENGTH_MS * PE_MAX_FS_KHZ )
#define PE_MAX_FRAME_LENGTH_ST_1    ( PE_MAX_FRAME_LENGTH >> 2 )
#define PE_MAX_FRAME_LENGTH_ST_2    ( PE_MAX_FRAME_LENGTH >> 1 )
#define PE_MAX_LAG_MS               18
#define PE_MIN_LAG_MS               2
#define PE_MAX_LAG                  ( PE_MAX_LAG_MS * PE_MAX_FS_KHZ )
#define PE_MIN_LAG                  ( PE_MIN_LAG_MS * PE_MAX_FS_KHZ )
#define PE_D_SRCH_LENGTH            24
#define PE_NB_STAGE3_LAGS           5
#define PE_NB_CBKS_STAGE2           3
#define PE_NB_CBKS_STAGE2_EXT       11
#define PE_NB_CBKS_STAGE3_MAX       34
#define PE_NB_CBKS_STAGE3_MID       24
#define PE_NB_CBKS_STAGE3_MIN       16
#define PE_NB_CBKS_STAGE3_10MS      12
#define PE_NB_CBKS_STAGE2_10MS      3
#define PE_SHORTLAG_BIAS            0.2f
#define PE_PREVLAG_BIAS             0.2f
#define PE_FLATCONTOUR_BIAS         0.05f
#define SILK_PE_MIN_COMPLEX         0
#define SILK_PE_MID_COMPLEX         1
#define SILK_PE_MAX_COMPLEX         2
static const opus_int8 silk_CB_lags_stage2[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE2_EXT ];
static const opus_int8 silk_CB_lags_stage3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ];
static const opus_int8 silk_Lag_range_stage3[ SILK_PE_MAX_COMPLEX + 1 ] [ PE_MAX_NB_SUBFR ][ 2 ];
static const opus_int8 silk_nb_cbk_searchs_stage3[ SILK_PE_MAX_COMPLEX + 1 ];
static const opus_int8 silk_CB_lags_stage2_10_ms[ PE_MAX_NB_SUBFR >> 1][ 3 ];
static const opus_int8 silk_CB_lags_stage3_10_ms[ PE_MAX_NB_SUBFR >> 1 ][ 12 ];
static const opus_int8 silk_Lag_range_stage3_10_ms[ PE_MAX_NB_SUBFR >> 1 ][ 2 ];

/* silk/resampler_rom.h, resampler_rom.c (the coefficients used here) */
#define RESAMPLER_ORDER_FIR_12                  8
static const opus_int16 silk_resampler_down2_0 = 9872;
static const opus_int16 silk_resampler_down2_1 = 39809 - 65536;
static const opus_int16 silk_Resampler_2_3_COEFS_LQ[ 2 + 2 * 2 ] = {
     -2797,  -6507,
      4697,  10739,
      1567,   8276,
};
/* silk/resampler_private.h */
#define RESAMPLER_MAX_BATCH_SIZE_MS             10
#define RESAMPLER_MAX_FS_KHZ                    48
#define RESAMPLER_MAX_BATCH_SIZE_IN             ( RESAMPLER_MAX_BATCH_SIZE_MS * RESAMPLER_MAX_FS_KHZ )
static void silk_resampler_private_AR2( opus_int32 S[], opus_int32 out_Q8[], const opus_int16 in[], const opus_int16 A_Q14[], opus_int32 len );
static void silk_resampler_down2( opus_int32 *S, opus_int16 *out, const opus_int16 *in, opus_int32 inLen );
static void silk_resampler_down2_3( opus_int32 *S, opus_int16 *out, const opus_int16 *in, opus_int32 inLen );

/* silk/structs.h: VAD state */
typedef struct {
    opus_int32                  AnaState[ 2 ];                  /* Analysis filterbank state: 0-8 kHz                                   */
    opus_int32                  AnaState1[ 2 ];                 /* Analysis filterbank state: 0-4 kHz                                   */
    opus_int32                  AnaState2[ 2 ];                 /* Analysis filterbank state: 0-2 kHz                                   */
    opus_int32                  XnrgSubfr[ VAD_N_BANDS ];       /* Subframe energies                                                    */
    opus_int32                  NrgRatioSmth_Q8[ VAD_N_BANDS ]; /* Smoothed energy level in each band                                   */
    opus_int16                  HPstate;                        /* State of differentiator in the lowest band                           */
    opus_int32                  NL[ VAD_N_BANDS ];              /* Noise energy level in each band                                      */
    opus_int32                  inv_NL[ VAD_N_BANDS ];          /* Inverse noise energy level in each band                              */
    opus_int32                  NoiseLevelBias[ VAD_N_BANDS ];  /* Noise level estimator bias/offset                                    */
    opus_int32                  counter;                        /* Frame counter used in the initial phase                              */
} silk_VAD_state;

/* the parts of silk_encoder_state (silk/structs.h) the VAD uses */
typedef struct {
    opus_int                    fs_kHz;
    opus_int                    frame_length;
    silk_VAD_state              sVAD;
    opus_int                    speech_activity_Q8;
    opus_int                    input_tilt_Q15;
    opus_int                    input_quality_bands_Q15[ VAD_N_BANDS ];
} silk_encoder_state;

static void silk_ana_filt_bank_1( const opus_int16 *in, opus_int32 *S, opus_int16 *outL, opus_int16 *outH, const opus_int32 N );

/* prototypes of the float functions used before they're defined */
static void silk_autocorrelation_FLP( silk_float *results, const silk_float *inputData, opus_int inputDataSize, opus_int correlationCount, int arch );
static silk_float silk_schur_FLP( silk_float refl_coef[], const silk_float auto_corr[], opus_int order );
static void silk_k2a_FLP( silk_float *A, const silk_float *rc, opus_int32 order );
static void silk_bwexpander_FLP( silk_float *ar, const opus_int d, const silk_float chirp );
static void silk_apply_sine_window_FLP( silk_float px_win[], const silk_float px[], const opus_int win_type, const opus_int length );
static void silk_LPC_analysis_filter_FLP( silk_float r_LPC[], const silk_float PredCoef[], const silk_float s[], const opus_int length, const opus_int Order );
static opus_int silk_pitch_analysis_core_FLP( const silk_float *frame, opus_int *pitch_out, opus_int16 *lagIndex, opus_int8 *contourIndex,
    silk_float *LTPCorr, opus_int prevLag, const silk_float search_thres1, const silk_float search_thres2,
    const opus_int Fs_kHz, const opus_int complexity, const opus_int nb_subfr, int arch );

/*=====================================================================
  celt/pitch.c: celt_pitch_xcorr_c(), float build (its simple version;
  the unrolled one computes the same sums in the same order)
  =====================================================================*/
static void celt_pitch_xcorr( const opus_val16 *_x, const opus_val16 *_y,
      opus_val32 *xcorr, int len, int max_pitch, int arch )
{
   int i, j;
   (void)arch;
   for (i=0;i<max_pitch;i++)
   {
      opus_val32 sum = 0;
      for (j=0;j<len;j++)
         sum = MAC16_16(sum, _x[j], _y[i+j]);
      xcorr[i] = sum;
   }
}

/*=====================================================================
  libopus code
  =====================================================================*/

/* ---- silk/float/energy_FLP.c ---- */
/* sum of squares of a silk_float array, with result as double */
static double silk_energy_FLP(
    const silk_float    *data,
    opus_int            dataSize
)
{
    opus_int i;
    double   result;

    /* 4x unrolled loop */
    result = 0.0;
    for( i = 0; i < dataSize - 3; i += 4 ) {
        result += data[ i + 0 ] * (double)data[ i + 0 ] +
                  data[ i + 1 ] * (double)data[ i + 1 ] +
                  data[ i + 2 ] * (double)data[ i + 2 ] +
                  data[ i + 3 ] * (double)data[ i + 3 ];
    }

    /* add any remaining products */
    for( ; i < dataSize; i++ ) {
        result += data[ i ] * (double)data[ i ];
    }

    silk_assert( result >= 0.0 );
    return result;
}

/* ---- silk/float/inner_product_FLP.c ---- */
/* inner product of two silk_float arrays, with result as double */
static double silk_inner_product_FLP_c(
    const silk_float    *data1,
    const silk_float    *data2,
    opus_int            dataSize
)
{
    opus_int i;
    double   result;

    /* 4x unrolled loop */
    result = 0.0;
    for( i = 0; i < dataSize - 3; i += 4 ) {
        result += data1[ i + 0 ] * (double)data2[ i + 0 ] +
                  data1[ i + 1 ] * (double)data2[ i + 1 ] +
                  data1[ i + 2 ] * (double)data2[ i + 2 ] +
                  data1[ i + 3 ] * (double)data2[ i + 3 ];
    }

    /* add any remaining products */
    for( ; i < dataSize; i++ ) {
        result += data1[ i ] * (double)data2[ i ];
    }

    return result;
}

/* ---- silk/float/autocorrelation_FLP.c ---- */
/* compute autocorrelation */
static void silk_autocorrelation_FLP(
    silk_float          *results,           /* O    result (length correlationCount)                            */
    const silk_float    *inputData,         /* I    input data to correlate                                     */
    opus_int            inputDataSize,      /* I    length of input                                             */
    opus_int            correlationCount,    /* I    number of correlation taps to compute                       */
    int                 arch
)
{
    opus_int i;

    if( correlationCount > inputDataSize ) {
        correlationCount = inputDataSize;
    }

    for( i = 0; i < correlationCount; i++ ) {
        results[ i ] =  (silk_float)silk_inner_product_FLP( inputData, inputData + i, inputDataSize - i, arch );
    }
}

/* ---- silk/float/schur_FLP.c ---- */
static silk_float silk_schur_FLP(                  /* O    returns residual energy                                     */
    silk_float          refl_coef[],        /* O    reflection coefficients (length order)                      */
    const silk_float    auto_corr[],        /* I    autocorrelation sequence (length order+1)                   */
    opus_int            order               /* I    order                                                       */
)
{
    opus_int   k, n;
    double C[ SILK_MAX_ORDER_LPC + 1 ][ 2 ];
    double Ctmp1, Ctmp2, rc_tmp;

    celt_assert( order >= 0 && order <= SILK_MAX_ORDER_LPC );

    /* Copy correlations */
    k = 0;
    do {
        C[ k ][ 0 ] = C[ k ][ 1 ] = auto_corr[ k ];
    } while( ++k <= order );

    for( k = 0; k < order; k++ ) {
        /* Get reflection coefficient */
        rc_tmp = -C[ k + 1 ][ 0 ] / silk_max_float( C[ 0 ][ 1 ], 1e-9f );

        /* Save the output */
        refl_coef[ k ] = (silk_float)rc_tmp;

        /* Update correlations */
        for( n = 0; n < order - k; n++ ) {
            Ctmp1 = C[ n + k + 1 ][ 0 ];
            Ctmp2 = C[ n ][ 1 ];
            C[ n + k + 1 ][ 0 ] = Ctmp1 + Ctmp2 * rc_tmp;
            C[ n ][ 1 ]         = Ctmp2 + Ctmp1 * rc_tmp;
        }
    }

    /* Return residual energy */
    return (silk_float)C[ 0 ][ 1 ];
}

/* ---- silk/float/k2a_FLP.c ---- */
/* step up function, converts reflection coefficients to prediction coefficients */
static void silk_k2a_FLP(
    silk_float          *A,                 /* O     prediction coefficients [order]                            */
    const silk_float    *rc,                /* I     reflection coefficients [order]                            */
    opus_int32          order               /* I     prediction order                                           */
)
{
    opus_int   k, n;
    silk_float rck, tmp1, tmp2;

    for( k = 0; k < order; k++ ) {
        rck = rc[ k ];
        for( n = 0; n < (k + 1) >> 1; n++ ) {
            tmp1 = A[ n ];
            tmp2 = A[ k - n - 1 ];
            A[ n ]         = tmp1 + tmp2 * rck;
            A[ k - n - 1 ] = tmp2 + tmp1 * rck;
        }
        A[ k ] = -rck;
    }
}

/* ---- silk/float/bwexpander_FLP.c ---- */
/* Chirp (bw expand) LP AR filter */
static void silk_bwexpander_FLP(
    silk_float          *ar,                /* I/O  AR filter to be expanded (without leading 1)                */
    const opus_int      d,                  /* I    length of ar                                                */
    const silk_float    chirp               /* I    chirp factor (typically in range (0..1) )                   */
)
{
    opus_int   i;
    silk_float cfac = chirp;

    for( i = 0; i < d - 1; i++ ) {
        ar[ i ] *=  cfac;
        cfac    *=  chirp;
    }
    ar[ d - 1 ] *=  cfac;
}

/* ---- silk/float/apply_sine_window_FLP.c ---- */
/* Apply sine window to signal vector   */
/* Window types:                        */
/*  1 -> sine window from 0 to pi/2     */
/*  2 -> sine window from pi/2 to pi    */
static void silk_apply_sine_window_FLP(
    silk_float                      px_win[],                           /* O    Pointer to windowed signal                  */
    const silk_float                px[],                               /* I    Pointer to input signal                     */
    const opus_int                  win_type,                           /* I    Selects a window type                       */
    const opus_int                  length                              /* I    Window length, multiple of 4                */
)
{
    opus_int   k;
    silk_float freq, c, S0, S1;

    celt_assert( win_type == 1 || win_type == 2 );

    /* Length must be multiple of 4 */
    celt_assert( ( length & 3 ) == 0 );

    freq = PI / ( length + 1 );

    /* Approximation of 2 * cos(f) */
    c = 2.0f - freq * freq;

    /* Initialize state */
    if( win_type < 2 ) {
        /* Start from 0 */
        S0 = 0.0f;
        /* Approximation of sin(f) */
        S1 = freq;
    } else {
        /* Start from 1 */
        S0 = 1.0f;
        /* Approximation of cos(f) */
        S1 = 0.5f * c;
    }

    /* Uses the recursive equation:   sin(n*f) = 2 * cos(f) * sin((n-1)*f) - sin((n-2)*f)   */
    /* 4 samples at a time */
    for( k = 0; k < length; k += 4 ) {
        px_win[ k + 0 ] = px[ k + 0 ] * 0.5f * ( S0 + S1 );
        px_win[ k + 1 ] = px[ k + 1 ] * S1;
        S0 = c * S1 - S0;
        px_win[ k + 2 ] = px[ k + 2 ] * 0.5f * ( S1 + S0 );
        px_win[ k + 3 ] = px[ k + 3 ] * S0;
        S1 = c * S0 - S1;
    }
}

/* ---- silk/float/LPC_analysis_filter_FLP.c ---- */
/************************************************/
/* LPC analysis filter                          */
/* NB! State is kept internally and the         */
/* filter always starts with zero state         */
/* first Order output samples are set to zero   */
/************************************************/

/* 16th order LPC analysis filter, does not write first 16 samples */
static OPUS_INLINE void silk_LPC_analysis_filter16_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 16; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ]  * PredCoef[ 0 ]  +
                   s_ptr[ -1 ]  * PredCoef[ 1 ]  +
                   s_ptr[ -2 ]  * PredCoef[ 2 ]  +
                   s_ptr[ -3 ]  * PredCoef[ 3 ]  +
                   s_ptr[ -4 ]  * PredCoef[ 4 ]  +
                   s_ptr[ -5 ]  * PredCoef[ 5 ]  +
                   s_ptr[ -6 ]  * PredCoef[ 6 ]  +
                   s_ptr[ -7 ]  * PredCoef[ 7 ]  +
                   s_ptr[ -8 ]  * PredCoef[ 8 ]  +
                   s_ptr[ -9 ]  * PredCoef[ 9 ]  +
                   s_ptr[ -10 ] * PredCoef[ 10 ] +
                   s_ptr[ -11 ] * PredCoef[ 11 ] +
                   s_ptr[ -12 ] * PredCoef[ 12 ] +
                   s_ptr[ -13 ] * PredCoef[ 13 ] +
                   s_ptr[ -14 ] * PredCoef[ 14 ] +
                   s_ptr[ -15 ] * PredCoef[ 15 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 12th order LPC analysis filter, does not write first 12 samples */
static OPUS_INLINE void silk_LPC_analysis_filter12_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 12; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ]  * PredCoef[ 0 ]  +
                   s_ptr[ -1 ]  * PredCoef[ 1 ]  +
                   s_ptr[ -2 ]  * PredCoef[ 2 ]  +
                   s_ptr[ -3 ]  * PredCoef[ 3 ]  +
                   s_ptr[ -4 ]  * PredCoef[ 4 ]  +
                   s_ptr[ -5 ]  * PredCoef[ 5 ]  +
                   s_ptr[ -6 ]  * PredCoef[ 6 ]  +
                   s_ptr[ -7 ]  * PredCoef[ 7 ]  +
                   s_ptr[ -8 ]  * PredCoef[ 8 ]  +
                   s_ptr[ -9 ]  * PredCoef[ 9 ]  +
                   s_ptr[ -10 ] * PredCoef[ 10 ] +
                   s_ptr[ -11 ] * PredCoef[ 11 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 10th order LPC analysis filter, does not write first 10 samples */
static OPUS_INLINE void silk_LPC_analysis_filter10_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 10; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ] * PredCoef[ 0 ]  +
                   s_ptr[ -1 ] * PredCoef[ 1 ]  +
                   s_ptr[ -2 ] * PredCoef[ 2 ]  +
                   s_ptr[ -3 ] * PredCoef[ 3 ]  +
                   s_ptr[ -4 ] * PredCoef[ 4 ]  +
                   s_ptr[ -5 ] * PredCoef[ 5 ]  +
                   s_ptr[ -6 ] * PredCoef[ 6 ]  +
                   s_ptr[ -7 ] * PredCoef[ 7 ]  +
                   s_ptr[ -8 ] * PredCoef[ 8 ]  +
                   s_ptr[ -9 ] * PredCoef[ 9 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 8th order LPC analysis filter, does not write first 8 samples */
static OPUS_INLINE void silk_LPC_analysis_filter8_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 8; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ] * PredCoef[ 0 ]  +
                   s_ptr[ -1 ] * PredCoef[ 1 ]  +
                   s_ptr[ -2 ] * PredCoef[ 2 ]  +
                   s_ptr[ -3 ] * PredCoef[ 3 ]  +
                   s_ptr[ -4 ] * PredCoef[ 4 ]  +
                   s_ptr[ -5 ] * PredCoef[ 5 ]  +
                   s_ptr[ -6 ] * PredCoef[ 6 ]  +
                   s_ptr[ -7 ] * PredCoef[ 7 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 6th order LPC analysis filter, does not write first 6 samples */
static OPUS_INLINE void silk_LPC_analysis_filter6_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 6; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ] * PredCoef[ 0 ]  +
                   s_ptr[ -1 ] * PredCoef[ 1 ]  +
                   s_ptr[ -2 ] * PredCoef[ 2 ]  +
                   s_ptr[ -3 ] * PredCoef[ 3 ]  +
                   s_ptr[ -4 ] * PredCoef[ 4 ]  +
                   s_ptr[ -5 ] * PredCoef[ 5 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/************************************************/
/* LPC analysis filter                          */
/* NB! State is kept internally and the         */
/* filter always starts with zero state         */
/* first Order output samples are set to zero   */
/************************************************/
static void silk_LPC_analysis_filter_FLP(
    silk_float                      r_LPC[],                            /* O    LPC residual signal                         */
    const silk_float                PredCoef[],                         /* I    LPC coefficients                            */
    const silk_float                s[],                                /* I    Input signal                                */
    const opus_int                  length,                             /* I    Length of input signal                      */
    const opus_int                  Order                               /* I    LPC order                                   */
)
{
    celt_assert( Order <= length );

    switch( Order ) {
        case 6:
            silk_LPC_analysis_filter6_FLP(  r_LPC, PredCoef, s, length );
        break;

        case 8:
            silk_LPC_analysis_filter8_FLP(  r_LPC, PredCoef, s, length );
        break;

        case 10:
            silk_LPC_analysis_filter10_FLP( r_LPC, PredCoef, s, length );
        break;

        case 12:
            silk_LPC_analysis_filter12_FLP( r_LPC, PredCoef, s, length );
        break;

        case 16:
            silk_LPC_analysis_filter16_FLP( r_LPC, PredCoef, s, length );
        break;

        default:
            celt_assert( 0 );
        break;
    }

    /* Set first Order output samples to zero */
    silk_memset( r_LPC, 0, Order * sizeof( silk_float ) );
}

/* ---- silk/float/sort_FLP.c ---- */
/* Insertion sort (fast for already almost sorted arrays):  */
/* Best case:  O(n)   for an already sorted array           */
/* Worst case: O(n^2) for an inversely sorted array         */




static void silk_insertion_sort_decreasing_FLP(
    silk_float          *a,                 /* I/O  Unsorted / Sorted vector                                    */
    opus_int            *idx,               /* O    Index vector for the sorted elements                        */
    const opus_int      L,                  /* I    Vector length                                               */
    const opus_int      K                   /* I    Number of correctly sorted positions                        */
)
{
    silk_float value;
    opus_int   i, j;

    /* Safety checks */
    celt_assert( K >  0 );
    celt_assert( L >  0 );
    celt_assert( L >= K );

    /* Write start indices in index vector */
    for( i = 0; i < K; i++ ) {
        idx[ i ] = i;
    }

    /* Sort vector elements by value, decreasing order */
    for( i = 1; i < K; i++ ) {
        value = a[ i ];
        for( j = i - 1; ( j >= 0 ) && ( value > a[ j ] ); j-- ) {
            a[ j + 1 ]   = a[ j ];      /* Shift value */
            idx[ j + 1 ] = idx[ j ];    /* Shift index */
        }
        a[ j + 1 ]   = value;   /* Write value */
        idx[ j + 1 ] = i;       /* Write index */
    }

    /* If less than L values are asked check the remaining values,      */
    /* but only spend CPU to ensure that the K first values are correct */
    for( i = K; i < L; i++ ) {
        value = a[ i ];
        if( value > a[ K - 1 ] ) {
            for( j = K - 2; ( j >= 0 ) && ( value > a[ j ] ); j-- ) {
                a[ j + 1 ]   = a[ j ];      /* Shift value */
                idx[ j + 1 ] = idx[ j ];    /* Shift index */
            }
            a[ j + 1 ]   = value;   /* Write value */
            idx[ j + 1 ] = i;       /* Write index */
        }
    }
}

/* ---- silk/float/burg_modified_FLP.c ---- */
#define MAX_FRAME_SIZE              384 /* subfr_length * nb_subfr = ( 0.005 * 16000 + 16 ) * 4 = 384*/

/* Compute reflection coefficients from input signal */
static silk_float silk_burg_modified_FLP(          /* O    returns residual energy                                     */
    silk_float          A[],                /* O    prediction coefficients (length order)                      */
    const silk_float    x[],                /* I    input signal, length: nb_subfr*(D+L_sub)                    */
    const silk_float    minInvGain,         /* I    minimum inverse prediction gain                             */
    const opus_int      subfr_length,       /* I    input signal subframe length (incl. D preceding samples)    */
    const opus_int      nb_subfr,           /* I    number of subframes stacked in x                            */
    const opus_int      D,                  /* I    order                                                       */
    int                 arch
)
{
    opus_int         k, n, s, reached_max_gain;
    double           C0, invGain, num, nrg_f, nrg_b, rc, Atmp, tmp1, tmp2;
    const silk_float *x_ptr;
    double           C_first_row[ SILK_MAX_ORDER_LPC ], C_last_row[ SILK_MAX_ORDER_LPC ];
    double           CAf[ SILK_MAX_ORDER_LPC + 1 ], CAb[ SILK_MAX_ORDER_LPC + 1 ];
    double           Af[ SILK_MAX_ORDER_LPC ];

    celt_assert( subfr_length * nb_subfr <= MAX_FRAME_SIZE );

    /* Compute autocorrelations, added over subframes */
    C0 = silk_energy_FLP( x, nb_subfr * subfr_length );
    silk_memset( C_first_row, 0, SILK_MAX_ORDER_LPC * sizeof( double ) );
    for( s = 0; s < nb_subfr; s++ ) {
        x_ptr = x + s * subfr_length;
        for( n = 1; n < D + 1; n++ ) {
            C_first_row[ n - 1 ] += silk_inner_product_FLP( x_ptr, x_ptr + n, subfr_length - n, arch );
        }
    }
    silk_memcpy( C_last_row, C_first_row, SILK_MAX_ORDER_LPC * sizeof( double ) );

    /* Initialize. Make sure the noise corresponds to at least a unit pulse. */
    CAb[ 0 ] = CAf[ 0 ] = C0 + FIND_LPC_COND_FAC * C0 + 1e-1;
    invGain = 1.0f;
    reached_max_gain = 0;
    for( n = 0; n < D; n++ ) {
        /* Update first row of correlation matrix (without first element) */
        /* Update last row of correlation matrix (without last element, stored in reversed order) */
        /* Update C * Af */
        /* Update C * flipud(Af) (stored in reversed order) */
        for( s = 0; s < nb_subfr; s++ ) {
            x_ptr = x + s * subfr_length;
            tmp1 = x_ptr[ n ];
            tmp2 = x_ptr[ subfr_length - n - 1 ];
            for( k = 0; k < n; k++ ) {
                C_first_row[ k ] -= x_ptr[ n ] * x_ptr[ n - k - 1 ];
                C_last_row[ k ]  -= x_ptr[ subfr_length - n - 1 ] * x_ptr[ subfr_length - n + k ];
                Atmp = Af[ k ];
                tmp1 += x_ptr[ n - k - 1 ] * Atmp;
                tmp2 += x_ptr[ subfr_length - n + k ] * Atmp;
            }
            for( k = 0; k <= n; k++ ) {
                CAf[ k ] -= tmp1 * x_ptr[ n - k ];
                CAb[ k ] -= tmp2 * x_ptr[ subfr_length - n + k - 1 ];
            }
        }
        tmp1 = C_first_row[ n ];
        tmp2 = C_last_row[ n ];
        for( k = 0; k < n; k++ ) {
            Atmp = Af[ k ];
            tmp1 += C_last_row[  n - k - 1 ] * Atmp;
            tmp2 += C_first_row[ n - k - 1 ] * Atmp;
        }
        CAf[ n + 1 ] = tmp1;
        CAb[ n + 1 ] = tmp2;

        /* Calculate nominator and denominator for the next order reflection (parcor) coefficient */
        num = CAb[ n + 1 ];
        nrg_b = CAb[ 0 ];
        nrg_f = CAf[ 0 ];
        for( k = 0; k < n; k++ ) {
            Atmp = Af[ k ];
            num   += CAb[ n - k ] * Atmp;
            nrg_b += CAb[ k + 1 ] * Atmp;
            nrg_f += CAf[ k + 1 ] * Atmp;
        }
        silk_assert( nrg_f > 0.0 );
        silk_assert( nrg_b > 0.0 );

        /* Calculate the next order reflection (parcor) coefficient */
        rc = -2.0 * num / ( nrg_f + nrg_b );
        silk_assert( rc > -1.0 && rc < 1.0 );

        /* Update inverse prediction gain */
        tmp1 = invGain * ( 1.0 - rc * rc );
        if( tmp1 <= minInvGain ) {
            /* Max prediction gain exceeded; set reflection coefficient such that max prediction gain is exactly hit */
            rc = sqrt( 1.0 - minInvGain / invGain );
            if( num > 0 ) {
                /* Ensure adjusted reflection coefficients has the original sign */
                rc = -rc;
            }
            invGain = minInvGain;
            reached_max_gain = 1;
        } else {
            invGain = tmp1;
        }

        /* Update the AR coefficients */
        for( k = 0; k < (n + 1) >> 1; k++ ) {
            tmp1 = Af[ k ];
            tmp2 = Af[ n - k - 1 ];
            Af[ k ]         = tmp1 + rc * tmp2;
            Af[ n - k - 1 ] = tmp2 + rc * tmp1;
        }
        Af[ n ] = rc;

        if( reached_max_gain ) {
            /* Reached max prediction gain; set remaining coefficients to zero and exit loop */
            for( k = n + 1; k < D; k++ ) {
                Af[ k ] = 0.0;
            }
            break;
        }

        /* Update C * Af and C * Ab */
        for( k = 0; k <= n + 1; k++ ) {
            tmp1 = CAf[ k ];
            CAf[ k ]          += rc * CAb[ n - k + 1 ];
            CAb[ n - k + 1  ] += rc * tmp1;
        }
    }

    if( reached_max_gain ) {
        /* Convert to silk_float */
        for( k = 0; k < D; k++ ) {
            A[ k ] = (silk_float)( -Af[ k ] );
        }
        /* Subtract energy of preceding samples from C0 */
        for( s = 0; s < nb_subfr; s++ ) {
            C0 -= silk_energy_FLP( x + s * subfr_length, D );
        }
        /* Approximate residual energy */
        nrg_f = C0 * invGain;
    } else {
        /* Compute residual energy and store coefficients as silk_float */
        nrg_f = CAf[ 0 ];
        tmp1 = 1.0;
        for( k = 0; k < D; k++ ) {
            Atmp = Af[ k ];
            nrg_f += CAf[ k + 1 ] * Atmp;
            tmp1  += Atmp * Atmp;
            A[ k ] = (silk_float)(-Atmp);
        }
        nrg_f -= FIND_LPC_COND_FAC * C0 * tmp1;
    }

    /* Return residual energy */
    return (silk_float)nrg_f;
}

/* ---- silk/pitch_est_tables.c ---- */
static const opus_int8 silk_CB_lags_stage2_10_ms[ PE_MAX_NB_SUBFR >> 1][ PE_NB_CBKS_STAGE2_10MS ] =
{
    {0, 1, 0},
    {0, 0, 1}
};

static const opus_int8 silk_CB_lags_stage3_10_ms[ PE_MAX_NB_SUBFR >> 1 ][ PE_NB_CBKS_STAGE3_10MS ] =
{
    { 0, 0, 1,-1, 1,-1, 2,-2, 2,-2, 3,-3},
    { 0, 1, 0, 1,-1, 2,-1, 2,-2, 3,-2, 3}
};

static const opus_int8 silk_Lag_range_stage3_10_ms[ PE_MAX_NB_SUBFR >> 1 ][ 2 ] =
{
    {-3, 7},
    {-2, 7}
};

static const opus_int8 silk_CB_lags_stage2[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE2_EXT ] =
{
    {0, 2,-1,-1,-1, 0, 0, 1, 1, 0, 1},
    {0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
    {0,-1, 2, 1, 0, 1, 1, 0, 0,-1,-1}
};

static const opus_int8 silk_CB_lags_stage3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ] =
{
    {0, 0, 1,-1, 0, 1,-1, 0,-1, 1,-2, 2,-2,-2, 2,-3, 2, 3,-3,-4, 3,-4, 4, 4,-5, 5,-6,-5, 6,-7, 6, 5, 8,-9},
    {0, 0, 1, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 1,-1, 0, 1,-1,-1, 1,-1, 2, 1,-1, 2,-2,-2, 2,-2, 2, 2, 3,-3},
    {0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,-1, 1, 0, 0, 2, 1,-1, 2,-1,-1, 2,-1, 2, 2,-1, 3,-2,-2,-2, 3},
    {0, 1, 0, 0, 1, 0, 1,-1, 2,-1, 2,-1, 2, 3,-2, 3,-2,-2, 4, 4,-3, 5,-3,-4, 6,-4, 6, 5,-5, 8,-6,-5,-7, 9}
};

static const opus_int8 silk_Lag_range_stage3[ SILK_PE_MAX_COMPLEX + 1 ] [ PE_MAX_NB_SUBFR ][ 2 ] =
{
    /* Lags to search for low number of stage3 cbks */
    {
        {-5,8},
        {-1,6},
        {-1,6},
        {-4,10}
    },
    /* Lags to search for middle number of stage3 cbks */
    {
        {-6,10},
        {-2,6},
        {-1,6},
        {-5,10}
    },
    /* Lags to search for max number of stage3 cbks */
    {
        {-9,12},
        {-3,7},
        {-2,7},
        {-7,13}
    }
};

static const opus_int8 silk_nb_cbk_searchs_stage3[ SILK_PE_MAX_COMPLEX + 1 ] =
{
    PE_NB_CBKS_STAGE3_MIN,
    PE_NB_CBKS_STAGE3_MID,
    PE_NB_CBKS_STAGE3_MAX
};

/* ---- silk/resampler_private_AR2.c ---- */
/* Second order AR filter with single delay elements */
static void silk_resampler_private_AR2(
    opus_int32                      S[],            /* I/O  State vector [ 2 ]          */
    opus_int32                      out_Q8[],       /* O    Output signal               */
    const opus_int16                in[],           /* I    Input signal                */
    const opus_int16                A_Q14[],        /* I    AR coefficients, Q14        */
    opus_int32                      len             /* I    Signal length               */
)
{
    opus_int32    k;
    opus_int32    out32;

    for( k = 0; k < len; k++ ) {
        out32       = silk_ADD_LSHIFT32( S[ 0 ], (opus_int32)in[ k ], 8 );
        out_Q8[ k ] = out32;
        out32       = silk_LSHIFT( out32, 2 );
        S[ 0 ]      = silk_SMLAWB( S[ 1 ], out32, A_Q14[ 0 ] );
        S[ 1 ]      = silk_SMULWB( out32, A_Q14[ 1 ] );
    }
}

/* ---- silk/resampler_down2.c ---- */
/* Downsample by a factor 2 */
static void silk_resampler_down2(
    opus_int32                  *S,                 /* I/O  State vector [ 2 ]                                          */
    opus_int16                  *out,               /* O    Output signal [ floor(len/2) ]                              */
    const opus_int16            *in,                /* I    Input signal [ len ]                                        */
    opus_int32                  inLen               /* I    Number of input samples                                     */
)
{
    opus_int32 k, len2 = silk_RSHIFT32( inLen, 1 );
    opus_int32 in32, out32, Y, X;

    celt_assert( silk_resampler_down2_0 > 0 );
    celt_assert( silk_resampler_down2_1 < 0 );

    /* Internal variables and state are in Q10 format */
    for( k = 0; k < len2; k++ ) {
        /* Convert to Q10 */
        in32 = silk_LSHIFT( (opus_int32)in[ 2 * k ], 10 );

        /* All-pass section for even input sample */
        Y      = silk_SUB32( in32, S[ 0 ] );
        X      = silk_SMLAWB( Y, Y, silk_resampler_down2_1 );
        out32  = silk_ADD32( S[ 0 ], X );
        S[ 0 ] = silk_ADD32( in32, X );

        /* Convert to Q10 */
        in32 = silk_LSHIFT( (opus_int32)in[ 2 * k + 1 ], 10 );

        /* All-pass section for odd input sample, and add to output of previous section */
        Y      = silk_SUB32( in32, S[ 1 ] );
        X      = silk_SMULWB( Y, silk_resampler_down2_0 );
        out32  = silk_ADD32( out32, S[ 1 ] );
        out32  = silk_ADD32( out32, X );
        S[ 1 ] = silk_ADD32( in32, X );

        /* Add, convert back to int16 and store to output */
        out[ k ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( out32, 11 ) );
    }
}

/* ---- silk/resampler_down2_3.c ---- */
#define ORDER_FIR                   4

/* Downsample by a factor 2/3, low quality */
static void silk_resampler_down2_3(
    opus_int32                  *S,                 /* I/O  State vector [ 6 ]                                          */
    opus_int16                  *out,               /* O    Output signal [ floor(2*inLen/3) ]                          */
    const opus_int16            *in,                /* I    Input signal [ inLen ]                                      */
    opus_int32                  inLen               /* I    Number of input samples                                     */
)
{
    opus_int32 nSamplesIn, counter, res_Q6;
    VARDECL( opus_int32, buf );
    opus_int32 *buf_ptr;
    SAVE_STACK;

    ALLOC( buf, RESAMPLER_MAX_BATCH_SIZE_IN + ORDER_FIR, opus_int32 );

    /* Copy buffered samples to start of buffer */
    silk_memcpy( buf, S, ORDER_FIR * sizeof( opus_int32 ) );

    /* Iterate over blocks of frameSizeIn input samples */
    while( 1 ) {
        nSamplesIn = silk_min( inLen, RESAMPLER_MAX_BATCH_SIZE_IN );

        /* Second-order AR filter (output in Q8) */
        silk_resampler_private_AR2( &S[ ORDER_FIR ], &buf[ ORDER_FIR ], in,
            silk_Resampler_2_3_COEFS_LQ, nSamplesIn );

        /* Interpolate filtered signal */
        buf_ptr = buf;
        counter = nSamplesIn;
        while( counter > 2 ) {
            /* Inner product */
            res_Q6 = silk_SMULWB(         buf_ptr[ 0 ], silk_Resampler_2_3_COEFS_LQ[ 2 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 1 ], silk_Resampler_2_3_COEFS_LQ[ 3 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 2 ], silk_Resampler_2_3_COEFS_LQ[ 5 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 3 ], silk_Resampler_2_3_COEFS_LQ[ 4 ] );

            /* Scale down, saturate and store in output array */
            *out++ = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( res_Q6, 6 ) );

            res_Q6 = silk_SMULWB(         buf_ptr[ 1 ], silk_Resampler_2_3_COEFS_LQ[ 4 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 2 ], silk_Resampler_2_3_COEFS_LQ[ 5 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 3 ], silk_Resampler_2_3_COEFS_LQ[ 3 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 4 ], silk_Resampler_2_3_COEFS_LQ[ 2 ] );

            /* Scale down, saturate and store in output array */
            *out++ = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( res_Q6, 6 ) );

            buf_ptr += 3;
            counter -= 3;
        }

        in += nSamplesIn;
        inLen -= nSamplesIn;

        if( inLen > 0 ) {
            /* More iterations to do; copy last part of filtered signal to beginning of buffer */
            silk_memcpy( buf, &buf[ nSamplesIn ], ORDER_FIR * sizeof( opus_int32 ) );
        } else {
            break;
        }
    }

    /* Copy last part of filtered signal to the state for the next call */
    silk_memcpy( S, &buf[ nSamplesIn ], ORDER_FIR * sizeof( opus_int32 ) );
    RESTORE_STACK;
}

/* ---- silk/float/pitch_analysis_core_FLP.c ---- */
/*****************************************************************************
* Pitch analyser function
******************************************************************************/





#define SCRATCH_SIZE        22

/************************************************************/
/* Internally used functions                                */
/************************************************************/
static void silk_P_Ana_calc_corr_st3(
    silk_float cross_corr_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ], /* O 3 DIM correlation array */
    const silk_float    frame[],            /* I vector to correlate                                            */
    opus_int            start_lag,          /* I start lag                                                      */
    opus_int            sf_length,          /* I sub frame length                                               */
    opus_int            nb_subfr,           /* I number of subframes                                            */
    opus_int            complexity,         /* I Complexity setting                                             */
    int                 arch                /* I Run-time architecture                                          */
);

static void silk_P_Ana_calc_energy_st3(
    silk_float energies_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ], /* O 3 DIM correlation array */
    const silk_float    frame[],            /* I vector to correlate                                            */
    opus_int            start_lag,          /* I start lag                                                      */
    opus_int            sf_length,          /* I sub frame length                                               */
    opus_int            nb_subfr,           /* I number of subframes                                            */
    opus_int            complexity          /* I Complexity setting                                             */
);

/************************************************************/
/* CORE PITCH ANALYSIS FUNCTION                             */
/************************************************************/
static opus_int silk_pitch_analysis_core_FLP(      /* O    Voicing estimate: 0 voiced, 1 unvoiced                      */
    const silk_float    *frame,             /* I    Signal of length PE_FRAME_LENGTH_MS*Fs_kHz                  */
    opus_int            *pitch_out,         /* O    Pitch lag values [nb_subfr]                                 */
    opus_int16          *lagIndex,          /* O    Lag Index                                                   */
    opus_int8           *contourIndex,      /* O    Pitch contour Index                                         */
    silk_float          *LTPCorr,           /* I/O  Normalized correlation; input: value from previous frame    */
    opus_int            prevLag,            /* I    Last lag of previous frame; set to zero is unvoiced         */
    const silk_float    search_thres1,      /* I    First stage threshold for lag candidates 0 - 1              */
    const silk_float    search_thres2,      /* I    Final threshold for lag candidates 0 - 1                    */
    const opus_int      Fs_kHz,             /* I    sample frequency (kHz)                                      */
    const opus_int      complexity,         /* I    Complexity setting, 0-2, where 2 is highest                 */
    const opus_int      nb_subfr,           /* I    Number of 5 ms subframes                                    */
    int                 arch                /* I    Run-time architecture                                       */
)
{
    opus_int   i, k, d, j;
    silk_float frame_8kHz[  PE_MAX_FRAME_LENGTH_MS * 8 ];
    silk_float frame_4kHz[  PE_MAX_FRAME_LENGTH_MS * 4 ];
    opus_int16 frame_8_FIX[ PE_MAX_FRAME_LENGTH_MS * 8 ];
    opus_int16 frame_4_FIX[ PE_MAX_FRAME_LENGTH_MS * 4 ];
    opus_int32 filt_state[ 6 ];
    silk_float threshold, contour_bias;
    silk_float C[ PE_MAX_NB_SUBFR][ (PE_MAX_LAG >> 1) + 5 ];
    opus_val32 xcorr[ PE_MAX_LAG_MS * 4 - PE_MIN_LAG_MS * 4 + 1 ];
    silk_float CC[ PE_NB_CBKS_STAGE2_EXT ];
    const silk_float *target_ptr, *basis_ptr;
    double    cross_corr, normalizer, energy, energy_tmp;
    opus_int   d_srch[ PE_D_SRCH_LENGTH ];
    opus_int16 d_comp[ (PE_MAX_LAG >> 1) + 5 ];
    opus_int   length_d_srch, length_d_comp;
    silk_float Cmax, CCmax, CCmax_b, CCmax_new_b, CCmax_new;
    opus_int   CBimax, CBimax_new, lag, start_lag, end_lag, lag_new;
    opus_int   cbk_size;
    silk_float lag_log2, prevLag_log2, delta_lag_log2_sqr;
    silk_float energies_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ];
    silk_float cross_corr_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ];
    opus_int   lag_counter;
    opus_int   frame_length, frame_length_8kHz, frame_length_4kHz;
    opus_int   sf_length, sf_length_8kHz, sf_length_4kHz;
    opus_int   min_lag, min_lag_8kHz, min_lag_4kHz;
    opus_int   max_lag, max_lag_8kHz, max_lag_4kHz;
    opus_int   nb_cbk_search;
    const opus_int8 *Lag_CB_ptr;

    /* Check for valid sampling frequency */
    celt_assert( Fs_kHz == 8 || Fs_kHz == 12 || Fs_kHz == 16 );

    /* Check for valid complexity setting */
    celt_assert( complexity >= SILK_PE_MIN_COMPLEX );
    celt_assert( complexity <= SILK_PE_MAX_COMPLEX );

    silk_assert( search_thres1 >= 0.0f && search_thres1 <= 1.0f );
    silk_assert( search_thres2 >= 0.0f && search_thres2 <= 1.0f );

    /* Set up frame lengths max / min lag for the sampling frequency */
    frame_length      = ( PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS ) * Fs_kHz;
    frame_length_4kHz = ( PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS ) * 4;
    frame_length_8kHz = ( PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS ) * 8;
    sf_length         = PE_SUBFR_LENGTH_MS * Fs_kHz;
    sf_length_4kHz    = PE_SUBFR_LENGTH_MS * 4;
    sf_length_8kHz    = PE_SUBFR_LENGTH_MS * 8;
    min_lag           = PE_MIN_LAG_MS * Fs_kHz;
    min_lag_4kHz      = PE_MIN_LAG_MS * 4;
    min_lag_8kHz      = PE_MIN_LAG_MS * 8;
    max_lag           = PE_MAX_LAG_MS * Fs_kHz - 1;
    max_lag_4kHz      = PE_MAX_LAG_MS * 4;
    max_lag_8kHz      = PE_MAX_LAG_MS * 8 - 1;

    /* Resample from input sampled at Fs_kHz to 8 kHz */
    if( Fs_kHz == 16 ) {
        /* Resample to 16 -> 8 khz */
        opus_int16 frame_16_FIX[ 16 * PE_MAX_FRAME_LENGTH_MS ];
        silk_float2short_array( frame_16_FIX, frame, frame_length );
        silk_memset( filt_state, 0, 2 * sizeof( opus_int32 ) );
        silk_resampler_down2( filt_state, frame_8_FIX, frame_16_FIX, frame_length );
        silk_short2float_array( frame_8kHz, frame_8_FIX, frame_length_8kHz );
    } else if( Fs_kHz == 12 ) {
        /* Resample to 12 -> 8 khz */
        opus_int16 frame_12_FIX[ 12 * PE_MAX_FRAME_LENGTH_MS ];
        silk_float2short_array( frame_12_FIX, frame, frame_length );
        silk_memset( filt_state, 0, 6 * sizeof( opus_int32 ) );
        silk_resampler_down2_3( filt_state, frame_8_FIX, frame_12_FIX, frame_length );
        silk_short2float_array( frame_8kHz, frame_8_FIX, frame_length_8kHz );
    } else {
        celt_assert( Fs_kHz == 8 );
        silk_float2short_array( frame_8_FIX, frame, frame_length_8kHz );
    }

    /* Decimate again to 4 kHz */
    silk_memset( filt_state, 0, 2 * sizeof( opus_int32 ) );
    silk_resampler_down2( filt_state, frame_4_FIX, frame_8_FIX, frame_length_8kHz );
    silk_short2float_array( frame_4kHz, frame_4_FIX, frame_length_4kHz );

    /* Low-pass filter */
    for( i = frame_length_4kHz - 1; i > 0; i-- ) {
        frame_4kHz[ i ] = silk_ADD_SAT16( frame_4kHz[ i ], frame_4kHz[ i - 1 ] );
    }

    /******************************************************************************
    * FIRST STAGE, operating in 4 khz
    ******************************************************************************/
    silk_memset(C, 0, sizeof(silk_float) * nb_subfr * ((PE_MAX_LAG >> 1) + 5));
    target_ptr = &frame_4kHz[ silk_LSHIFT( sf_length_4kHz, 2 ) ];
    for( k = 0; k < nb_subfr >> 1; k++ ) {
        /* Check that we are within range of the array */
        celt_assert( target_ptr >= frame_4kHz );
        celt_assert( target_ptr + sf_length_8kHz <= frame_4kHz + frame_length_4kHz );

        basis_ptr = target_ptr - min_lag_4kHz;

        /* Check that we are within range of the array */
        celt_assert( basis_ptr >= frame_4kHz );
        celt_assert( basis_ptr + sf_length_8kHz <= frame_4kHz + frame_length_4kHz );

        celt_pitch_xcorr( target_ptr, target_ptr-max_lag_4kHz, xcorr, sf_length_8kHz, max_lag_4kHz - min_lag_4kHz + 1, arch );

        /* Calculate first vector products before loop */
        cross_corr = xcorr[ max_lag_4kHz - min_lag_4kHz ];
        normalizer = silk_energy_FLP( target_ptr, sf_length_8kHz ) +
                     silk_energy_FLP( basis_ptr,  sf_length_8kHz ) +
                     sf_length_8kHz * 4000.0f;

        C[ 0 ][ min_lag_4kHz ] += (silk_float)( 2 * cross_corr / normalizer );

        /* From now on normalizer is computed recursively */
        for( d = min_lag_4kHz + 1; d <= max_lag_4kHz; d++ ) {
            basis_ptr--;

            /* Check that we are within range of the array */
            silk_assert( basis_ptr >= frame_4kHz );
            silk_assert( basis_ptr + sf_length_8kHz <= frame_4kHz + frame_length_4kHz );

            cross_corr = xcorr[ max_lag_4kHz - d ];

            /* Add contribution of new sample and remove contribution from oldest sample */
            normalizer +=
                basis_ptr[ 0 ] * (double)basis_ptr[ 0 ] -
                basis_ptr[ sf_length_8kHz ] * (double)basis_ptr[ sf_length_8kHz ];
            C[ 0 ][ d ] += (silk_float)( 2 * cross_corr / normalizer );
        }
        /* Update target pointer */
        target_ptr += sf_length_8kHz;
    }

    /* Apply short-lag bias */
    for( i = max_lag_4kHz; i >= min_lag_4kHz; i-- ) {
        C[ 0 ][ i ] -= C[ 0 ][ i ] * i / 4096.0f;
    }

    /* Sort */
    length_d_srch = 4 + 2 * complexity;
    celt_assert( 3 * length_d_srch <= PE_D_SRCH_LENGTH );
    silk_insertion_sort_decreasing_FLP( &C[ 0 ][ min_lag_4kHz ], d_srch, max_lag_4kHz - min_lag_4kHz + 1, length_d_srch );

    /* Escape if correlation is very low already here */
    Cmax = C[ 0 ][ min_lag_4kHz ];
    if( Cmax < 0.2f ) {
        silk_memset( pitch_out, 0, nb_subfr * sizeof( opus_int ) );
        *LTPCorr      = 0.0f;
        *lagIndex     = 0;
        *contourIndex = 0;
        return 1;
    }

    threshold = search_thres1 * Cmax;
    for( i = 0; i < length_d_srch; i++ ) {
        /* Convert to 8 kHz indices for the sorted correlation that exceeds the threshold */
        if( C[ 0 ][ min_lag_4kHz + i ] > threshold ) {
            d_srch[ i ] = silk_LSHIFT( d_srch[ i ] + min_lag_4kHz, 1 );
        } else {
            length_d_srch = i;
            break;
        }
    }
    celt_assert( length_d_srch > 0 );

    for( i = min_lag_8kHz - 5; i < max_lag_8kHz + 5; i++ ) {
        d_comp[ i ] = 0;
    }
    for( i = 0; i < length_d_srch; i++ ) {
        d_comp[ d_srch[ i ] ] = 1;
    }

    /* Convolution */
    for( i = max_lag_8kHz + 3; i >= min_lag_8kHz; i-- ) {
        d_comp[ i ] += d_comp[ i - 1 ] + d_comp[ i - 2 ];
    }

    length_d_srch = 0;
    for( i = min_lag_8kHz; i < max_lag_8kHz + 1; i++ ) {
        if( d_comp[ i + 1 ] > 0 ) {
            d_srch[ length_d_srch ] = i;
            length_d_srch++;
        }
    }

    /* Convolution */
    for( i = max_lag_8kHz + 3; i >= min_lag_8kHz; i-- ) {
        d_comp[ i ] += d_comp[ i - 1 ] + d_comp[ i - 2 ] + d_comp[ i - 3 ];
    }

    length_d_comp = 0;
    for( i = min_lag_8kHz; i < max_lag_8kHz + 4; i++ ) {
        if( d_comp[ i ] > 0 ) {
            d_comp[ length_d_comp ] = (opus_int16)( i - 2 );
            length_d_comp++;
        }
    }

    /**********************************************************************************
    ** SECOND STAGE, operating at 8 kHz, on lag sections with high correlation
    *************************************************************************************/
    /*********************************************************************************
    * Find energy of each subframe projected onto its history, for a range of delays
    *********************************************************************************/
    silk_memset( C, 0, PE_MAX_NB_SUBFR*((PE_MAX_LAG >> 1) + 5) * sizeof(silk_float));

    if( Fs_kHz == 8 ) {
        target_ptr = &frame[ PE_LTP_MEM_LENGTH_MS * 8 ];
    } else {
        target_ptr = &frame_8kHz[ PE_LTP_MEM_LENGTH_MS * 8 ];
    }
    for( k = 0; k < nb_subfr; k++ ) {
        energy_tmp = silk_energy_FLP( target_ptr, sf_length_8kHz ) + 1.0;
        for( j = 0; j < length_d_comp; j++ ) {
            d = d_comp[ j ];
            basis_ptr = target_ptr - d;
            cross_corr = silk_inner_product_FLP( basis_ptr, target_ptr, sf_length_8kHz, arch );
            if( cross_corr > 0.0f ) {
                energy = silk_energy_FLP( basis_ptr, sf_length_8kHz );
                C[ k ][ d ] = (silk_float)( 2 * cross_corr / ( energy + energy_tmp ) );
            } else {
                C[ k ][ d ] = 0.0f;
            }
        }
        target_ptr += sf_length_8kHz;
    }

    /* search over lag range and lags codebook */
    /* scale factor for lag codebook, as a function of center lag */

    CCmax   = 0.0f; /* This value doesn't matter */
    CCmax_b = -1000.0f;

    CBimax = 0; /* To avoid returning undefined lag values */
    lag = -1;   /* To check if lag with strong enough correlation has been found */

    if( prevLag > 0 ) {
        if( Fs_kHz == 12 ) {
            prevLag = silk_LSHIFT( prevLag, 1 ) / 3;
        } else if( Fs_kHz == 16 ) {
            prevLag = silk_RSHIFT( prevLag, 1 );
        }
        prevLag_log2 = silk_log2( (silk_float)prevLag );
    } else {
        prevLag_log2 = 0;
    }

    /* Set up stage 2 codebook based on number of subframes */
    if( nb_subfr == PE_MAX_NB_SUBFR ) {
        cbk_size   = PE_NB_CBKS_STAGE2_EXT;
        Lag_CB_ptr = &silk_CB_lags_stage2[ 0 ][ 0 ];
        if( Fs_kHz == 8 && complexity > SILK_PE_MIN_COMPLEX ) {
            /* If input is 8 khz use a larger codebook here because it is last stage */
            nb_cbk_search = PE_NB_CBKS_STAGE2_EXT;
        } else {
            nb_cbk_search = PE_NB_CBKS_STAGE2;
        }
    } else {
        cbk_size       = PE_NB_CBKS_STAGE2_10MS;
        Lag_CB_ptr     = &silk_CB_lags_stage2_10_ms[ 0 ][ 0 ];
        nb_cbk_search  = PE_NB_CBKS_STAGE2_10MS;
    }

    for( k = 0; k < length_d_srch; k++ ) {
        d = d_srch[ k ];
        for( j = 0; j < nb_cbk_search; j++ ) {
            CC[j] = 0.0f;
            for( i = 0; i < nb_subfr; i++ ) {
                /* Try all codebooks */
                CC[ j ] += C[ i ][ d + matrix_ptr( Lag_CB_ptr, i, j, cbk_size )];
            }
        }
        /* Find best codebook */
        CCmax_new  = -1000.0f;
        CBimax_new = 0;
        for( i = 0; i < nb_cbk_search; i++ ) {
            if( CC[ i ] > CCmax_new ) {
                CCmax_new = CC[ i ];
                CBimax_new = i;
            }
        }

        /* Bias towards shorter lags */
        lag_log2 = silk_log2( (silk_float)d );
        CCmax_new_b = CCmax_new - PE_SHORTLAG_BIAS * nb_subfr * lag_log2;

        /* Bias towards previous lag */
        if( prevLag > 0 ) {
            delta_lag_log2_sqr = lag_log2 - prevLag_log2;
            delta_lag_log2_sqr *= delta_lag_log2_sqr;
            CCmax_new_b -= PE_PREVLAG_BIAS * nb_subfr * (*LTPCorr) * delta_lag_log2_sqr / ( delta_lag_log2_sqr + 0.5f );
        }

        if( CCmax_new_b > CCmax_b &&                /* Find maximum biased correlation                  */
            CCmax_new > nb_subfr * search_thres2    /* Correlation needs to be high enough to be voiced */
        ) {
            CCmax_b = CCmax_new_b;
            CCmax   = CCmax_new;
            lag     = d;
            CBimax  = CBimax_new;
        }
    }

    if( lag == -1 ) {
        /* No suitable candidate found */
        silk_memset( pitch_out, 0, PE_MAX_NB_SUBFR * sizeof(opus_int) );
        *LTPCorr      = 0.0f;
        *lagIndex     = 0;
        *contourIndex = 0;
        return 1;
    }

    /* Output normalized correlation */
    *LTPCorr = (silk_float)( CCmax / nb_subfr );
    silk_assert( *LTPCorr >= 0.0f );

    if( Fs_kHz > 8 ) {
        /* Search in original signal */

        /* Compensate for decimation */
        silk_assert( lag == silk_SAT16( lag ) );
        if( Fs_kHz == 12 ) {
            lag = silk_RSHIFT_ROUND( silk_SMULBB( lag, 3 ), 1 );
        } else { /* Fs_kHz == 16 */
            lag = silk_LSHIFT( lag, 1 );
        }

        lag = silk_LIMIT_int( lag, min_lag, max_lag );
        start_lag = silk_max_int( lag - 2, min_lag );
        end_lag   = silk_min_int( lag + 2, max_lag );
        lag_new   = lag;                                    /* to avoid undefined lag */
        CBimax    = 0;                                      /* to avoid undefined lag */

        CCmax = -1000.0f;

        /* Calculate the correlations and energies needed in stage 3 */
        silk_P_Ana_calc_corr_st3( cross_corr_st3, frame, start_lag, sf_length, nb_subfr, complexity, arch );
        silk_P_Ana_calc_energy_st3( energies_st3, frame, start_lag, sf_length, nb_subfr, complexity );

        lag_counter = 0;
        silk_assert( lag == silk_SAT16( lag ) );
        contour_bias = PE_FLATCONTOUR_BIAS / lag;

        /* Set up cbk parameters according to complexity setting and frame length */
        if( nb_subfr == PE_MAX_NB_SUBFR ) {
            nb_cbk_search = (opus_int)silk_nb_cbk_searchs_stage3[ complexity ];
            cbk_size      = PE_NB_CBKS_STAGE3_MAX;
            Lag_CB_ptr    = &silk_CB_lags_stage3[ 0 ][ 0 ];
        } else {
            nb_cbk_search = PE_NB_CBKS_STAGE3_10MS;
            cbk_size      = PE_NB_CBKS_STAGE3_10MS;
            Lag_CB_ptr    = &silk_CB_lags_stage3_10_ms[ 0 ][ 0 ];
        }

        target_ptr = &frame[ PE_LTP_MEM_LENGTH_MS * Fs_kHz ];
        energy_tmp = silk_energy_FLP( target_ptr, nb_subfr * sf_length ) + 1.0;
        for( d = start_lag; d <= end_lag; d++ ) {
            for( j = 0; j < nb_cbk_search; j++ ) {
                cross_corr = 0.0;
                energy = energy_tmp;
                for( k = 0; k < nb_subfr; k++ ) {
                    cross_corr += cross_corr_st3[ k ][ j ][ lag_counter ];
                    energy     +=   energies_st3[ k ][ j ][ lag_counter ];
                }
                if( cross_corr > 0.0 ) {
                    CCmax_new = (silk_float)( 2 * cross_corr / energy );
                    /* Reduce depending on flatness of contour */
                    CCmax_new *= 1.0f - contour_bias * j;
                } else {
                    CCmax_new = 0.0f;
                }

                if( CCmax_new > CCmax && ( d + (opus_int)silk_CB_lags_stage3[ 0 ][ j ] ) <= max_lag ) {
                    CCmax   = CCmax_new;
                    lag_new = d;
                    CBimax  = j;
                }
            }
            lag_counter++;
        }

        for( k = 0; k < nb_subfr; k++ ) {
            pitch_out[ k ] = lag_new + matrix_ptr( Lag_CB_ptr, k, CBimax, cbk_size );
            pitch_out[ k ] = silk_LIMIT( pitch_out[ k ], min_lag, PE_MAX_LAG_MS * Fs_kHz );
        }
        *lagIndex = (opus_int16)( lag_new - min_lag );
        *contourIndex = (opus_int8)CBimax;
    } else {        /* Fs_kHz == 8 */
        /* Save Lags */
        for( k = 0; k < nb_subfr; k++ ) {
            pitch_out[ k ] = lag + matrix_ptr( Lag_CB_ptr, k, CBimax, cbk_size );
            pitch_out[ k ] = silk_LIMIT( pitch_out[ k ], min_lag_8kHz, PE_MAX_LAG_MS * 8 );
        }
        *lagIndex = (opus_int16)( lag - min_lag_8kHz );
        *contourIndex = (opus_int8)CBimax;
    }
    celt_assert( *lagIndex >= 0 );
    /* return as voiced */
    return 0;
}

/***********************************************************************
 * Calculates the correlations used in stage 3 search. In order to cover
 * the whole lag codebook for all the searched offset lags (lag +- 2),
 * the following correlations are needed in each sub frame:
 *
 * sf1: lag range [-8,...,7] total 16 correlations
 * sf2: lag range [-4,...,4] total 9 correlations
 * sf3: lag range [-3,....4] total 8 correltions
 * sf4: lag range [-6,....8] total 15 correlations
 *
 * In total 48 correlations. The direct implementation computed in worst
 * case 4*12*5 = 240 correlations, but more likely around 120.
 ***********************************************************************/
static void silk_P_Ana_calc_corr_st3(
    silk_float cross_corr_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ], /* O 3 DIM correlation array */
    const silk_float    frame[],            /* I vector to correlate                                            */
    opus_int            start_lag,          /* I start lag                                                      */
    opus_int            sf_length,          /* I sub frame length                                               */
    opus_int            nb_subfr,           /* I number of subframes                                            */
    opus_int            complexity,         /* I Complexity setting                                             */
    int                 arch                /* I Run-time architecture                                          */
)
{
    const silk_float *target_ptr;
    opus_int   i, j, k, lag_counter, lag_low, lag_high;
    opus_int   nb_cbk_search, delta, idx, cbk_size;
    silk_float scratch_mem[ SCRATCH_SIZE ];
    opus_val32 xcorr[ SCRATCH_SIZE ];
    const opus_int8 *Lag_range_ptr, *Lag_CB_ptr;

    celt_assert( complexity >= SILK_PE_MIN_COMPLEX );
    celt_assert( complexity <= SILK_PE_MAX_COMPLEX );

    if( nb_subfr == PE_MAX_NB_SUBFR ) {
        Lag_range_ptr = &silk_Lag_range_stage3[ complexity ][ 0 ][ 0 ];
        Lag_CB_ptr    = &silk_CB_lags_stage3[ 0 ][ 0 ];
        nb_cbk_search = silk_nb_cbk_searchs_stage3[ complexity ];
        cbk_size      = PE_NB_CBKS_STAGE3_MAX;
    } else {
        celt_assert( nb_subfr == PE_MAX_NB_SUBFR >> 1);
        Lag_range_ptr = &silk_Lag_range_stage3_10_ms[ 0 ][ 0 ];
        Lag_CB_ptr    = &silk_CB_lags_stage3_10_ms[ 0 ][ 0 ];
        nb_cbk_search = PE_NB_CBKS_STAGE3_10MS;
        cbk_size      = PE_NB_CBKS_STAGE3_10MS;
    }

    target_ptr = &frame[ silk_LSHIFT( sf_length, 2 ) ]; /* Pointer to middle of frame */
    for( k = 0; k < nb_subfr; k++ ) {
        lag_counter = 0;

        /* Calculate the correlations for each subframe */
        lag_low  = matrix_ptr( Lag_range_ptr, k, 0, 2 );
        lag_high = matrix_ptr( Lag_range_ptr, k, 1, 2 );
        silk_assert(lag_high-lag_low+1 <= SCRATCH_SIZE);
        celt_pitch_xcorr( target_ptr, target_ptr - start_lag - lag_high, xcorr, sf_length, lag_high - lag_low + 1, arch );
        for( j = lag_low; j <= lag_high; j++ ) {
            silk_assert( lag_counter < SCRATCH_SIZE );
            scratch_mem[ lag_counter ] = xcorr[ lag_high - j ];
            lag_counter++;
        }

        delta = matrix_ptr( Lag_range_ptr, k, 0, 2 );
        for( i = 0; i < nb_cbk_search; i++ ) {
            /* Fill out the 3 dim array that stores the correlations for */
            /* each code_book vector for each start lag */
            idx = matrix_ptr( Lag_CB_ptr, k, i, cbk_size ) - delta;
            for( j = 0; j < PE_NB_STAGE3_LAGS; j++ ) {
                silk_assert( idx + j < SCRATCH_SIZE );
                silk_assert( idx + j < lag_counter );
                cross_corr_st3[ k ][ i ][ j ] = scratch_mem[ idx + j ];
            }
        }
        target_ptr += sf_length;
    }
}

/********************************************************************/
/* Calculate the energies for first two subframes. The energies are */
/* calculated recursively.                                          */
/********************************************************************/
static void silk_P_Ana_calc_energy_st3(
    silk_float energies_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ], /* O 3 DIM correlation array */
    const silk_float    frame[],            /* I vector to correlate                                            */
    opus_int            start_lag,          /* I start lag                                                      */
    opus_int            sf_length,          /* I sub frame length                                               */
    opus_int            nb_subfr,           /* I number of subframes                                            */
    opus_int            complexity          /* I Complexity setting                                             */
)
{
    const silk_float *target_ptr, *basis_ptr;
    double    energy;
    opus_int   k, i, j, lag_counter;
    opus_int   nb_cbk_search, delta, idx, cbk_size, lag_diff;
    silk_float scratch_mem[ SCRATCH_SIZE ];
    const opus_int8 *Lag_range_ptr, *Lag_CB_ptr;

    celt_assert( complexity >= SILK_PE_MIN_COMPLEX );
    celt_assert( complexity <= SILK_PE_MAX_COMPLEX );

    if( nb_subfr == PE_MAX_NB_SUBFR ) {
        Lag_range_ptr = &silk_Lag_range_stage3[ complexity ][ 0 ][ 0 ];
        Lag_CB_ptr    = &silk_CB_lags_stage3[ 0 ][ 0 ];
        nb_cbk_search = silk_nb_cbk_searchs_stage3[ complexity ];
        cbk_size      = PE_NB_CBKS_STAGE3_MAX;
    } else {
        celt_assert( nb_subfr == PE_MAX_NB_SUBFR >> 1);
        Lag_range_ptr = &silk_Lag_range_stage3_10_ms[ 0 ][ 0 ];
        Lag_CB_ptr    = &silk_CB_lags_stage3_10_ms[ 0 ][ 0 ];
        nb_cbk_search = PE_NB_CBKS_STAGE3_10MS;
        cbk_size      = PE_NB_CBKS_STAGE3_10MS;
    }

    target_ptr = &frame[ silk_LSHIFT( sf_length, 2 ) ];
    for( k = 0; k < nb_subfr; k++ ) {
        lag_counter = 0;

        /* Calculate the energy for first lag */
        basis_ptr = target_ptr - ( start_lag + matrix_ptr( Lag_range_ptr, k, 0, 2 ) );
        energy = silk_energy_FLP( basis_ptr, sf_length ) + 1e-3;
        silk_assert( energy >= 0.0 );
        scratch_mem[lag_counter] = (silk_float)energy;
        lag_counter++;

        lag_diff = ( matrix_ptr( Lag_range_ptr, k, 1, 2 ) -  matrix_ptr( Lag_range_ptr, k, 0, 2 ) + 1 );
        for( i = 1; i < lag_diff; i++ ) {
            /* remove part outside new window */
            energy -= basis_ptr[sf_length - i] * (double)basis_ptr[sf_length - i];
            silk_assert( energy >= 0.0 );

            /* add part that comes into window */
            energy += basis_ptr[ -i ] * (double)basis_ptr[ -i ];
            silk_assert( energy >= 0.0 );
            silk_assert( lag_counter < SCRATCH_SIZE );
            scratch_mem[lag_counter] = (silk_float)energy;
            lag_counter++;
        }

        delta = matrix_ptr( Lag_range_ptr, k, 0, 2 );
        for( i = 0; i < nb_cbk_search; i++ ) {
            /* Fill out the 3 dim array that stores the correlations for    */
            /* each code_book vector for each start lag                     */
            idx = matrix_ptr( Lag_CB_ptr, k, i, cbk_size ) - delta;
            for( j = 0; j < PE_NB_STAGE3_LAGS; j++ ) {
                silk_assert( idx + j < SCRATCH_SIZE );
                silk_assert( idx + j < lag_counter );
                energies_st3[ k ][ i ][ j ] = scratch_mem[ idx + j ];
                silk_assert( energies_st3[ k ][ i ][ j ] >= 0.0f );
            }
        }
        target_ptr += sf_length;
    }
}

/* ---- silk/ana_filt_bank_1.c ---- */
/* Coefficients for 2-band filter bank based on first-order allpass filters */
static const opus_int16 A_fb1_20 = 5394 << 1;
static const opus_int16 A_fb1_21 = -24290; /* (opus_int16)(20623 << 1) */

/* Split signal into two decimated bands using first-order allpass filters */
static void silk_ana_filt_bank_1(
    const opus_int16            *in,                /* I    Input signal [N]                                            */
    opus_int32                  *S,                 /* I/O  State vector [2]                                            */
    opus_int16                  *outL,              /* O    Low band [N/2]                                              */
    opus_int16                  *outH,              /* O    High band [N/2]                                             */
    const opus_int32            N                   /* I    Number of input samples                                     */
)
{
    opus_int      k, N2 = silk_RSHIFT( N, 1 );
    opus_int32    in32, X, Y, out_1, out_2;

    /* Internal variables and state are in Q10 format */
    for( k = 0; k < N2; k++ ) {
        /* Convert to Q10 */
        in32 = silk_LSHIFT( (opus_int32)in[ 2 * k ], 10 );

        /* All-pass section for even input sample */
        Y      = silk_SUB32( in32, S[ 0 ] );
        X      = silk_SMLAWB( Y, Y, A_fb1_21 );
        out_1  = silk_ADD32( S[ 0 ], X );
        S[ 0 ] = silk_ADD32( in32, X );

        /* Convert to Q10 */
        in32 = silk_LSHIFT( (opus_int32)in[ 2 * k + 1 ], 10 );

        /* All-pass section for odd input sample, and add to output of previous section */
        Y      = silk_SUB32( in32, S[ 1 ] );
        X      = silk_SMULWB( Y, A_fb1_20 );
        out_2  = silk_ADD32( S[ 1 ], X );
        S[ 1 ] = silk_ADD32( in32, X );

        /* Add/subtract, convert back to int16 and store to output */
        outL[ k ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( silk_ADD32( out_2, out_1 ), 11 ) );
        outH[ k ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( silk_SUB32( out_2, out_1 ), 11 ) );
    }
}

/* ---- silk/sigm_Q15.c ---- */
/* Approximate sigmoid function */



/* fprintf(1, '%d, ', round(1024 * ([1 ./ (1 + exp(-(1:5))), 1] - 1 ./ (1 + exp(-(0:5)))))); */
static const opus_int32 sigm_LUT_slope_Q10[ 6 ] = {
    237, 153, 73, 30, 12, 7
};
/* fprintf(1, '%d, ', round(32767 * 1 ./ (1 + exp(-(0:5))))); */
static const opus_int32 sigm_LUT_pos_Q15[ 6 ] = {
    16384, 23955, 28861, 31213, 32178, 32548
};
/* fprintf(1, '%d, ', round(32767 * 1 ./ (1 + exp((0:5))))); */
static const opus_int32 sigm_LUT_neg_Q15[ 6 ] = {
    16384, 8812, 3906, 1554, 589, 219
};

static opus_int silk_sigm_Q15(
    opus_int                    in_Q5               /* I                                                                */
)
{
    opus_int ind;

    if( in_Q5 < 0 ) {
        /* Negative input */
        in_Q5 = -in_Q5;
        if( in_Q5 >= 6 * 32 ) {
            return 0;        /* Clip */
        } else {
            /* Linear interpolation of look up table */
            ind = silk_RSHIFT( in_Q5, 5 );
            return( sigm_LUT_neg_Q15[ ind ] - silk_SMULBB( sigm_LUT_slope_Q10[ ind ], in_Q5 & 0x1F ) );
        }
    } else {
        /* Positive input */
        if( in_Q5 >= 6 * 32 ) {
            return 32767;        /* clip */
        } else {
            /* Linear interpolation of look up table */
            ind = silk_RSHIFT( in_Q5, 5 );
            return( sigm_LUT_pos_Q15[ ind ] + silk_SMULBB( sigm_LUT_slope_Q10[ ind ], in_Q5 & 0x1F ) );
        }
    }
}

/* ---- silk/lin2log.c ---- */
/* Approximation of 128 * log2() (very close inverse of silk_log2lin()) */
/* Convert input to a log scale    */
static opus_int32 silk_lin2log(
    const opus_int32            inLin               /* I  input in linear scale                                         */
)
{
    opus_int32 lz, frac_Q7;

    silk_CLZ_FRAC( inLin, &lz, &frac_Q7 );

    /* Piece-wise parabolic approximation */
    return silk_ADD_LSHIFT32( silk_SMLAWB( frac_Q7, silk_MUL( frac_Q7, 128 - frac_Q7 ), 179 ), 31 - lz, 7 );
}

/* ---- silk/VAD.c ---- */
/* Silk VAD noise level estimation */
# if !defined(OPUS_X86_MAY_HAVE_SSE4_1)
static OPUS_INLINE void silk_VAD_GetNoiseLevels(
    const opus_int32             pX[ VAD_N_BANDS ], /* I    subband energies                            */
    silk_VAD_state              *psSilk_VAD         /* I/O  Pointer to Silk VAD state                   */
);
#endif

/**********************************/
/* Initialization of the Silk VAD */
/**********************************/
static opus_int silk_VAD_Init(                                         /* O    Return value, 0 if success                  */
    silk_VAD_state              *psSilk_VAD                     /* I/O  Pointer to Silk VAD state                   */
)
{
    opus_int b, ret = 0;

    /* reset state memory */
    silk_memset( psSilk_VAD, 0, sizeof( silk_VAD_state ) );

    /* init noise levels */
    /* Initialize array with approx pink noise levels (psd proportional to inverse of frequency) */
    for( b = 0; b < VAD_N_BANDS; b++ ) {
        psSilk_VAD->NoiseLevelBias[ b ] = silk_max_32( silk_DIV32_16( VAD_NOISE_LEVELS_BIAS, b + 1 ), 1 );
    }

    /* Initialize state */
    for( b = 0; b < VAD_N_BANDS; b++ ) {
        psSilk_VAD->NL[ b ]     = silk_MUL( 100, psSilk_VAD->NoiseLevelBias[ b ] );
        psSilk_VAD->inv_NL[ b ] = silk_DIV32( silk_int32_MAX, psSilk_VAD->NL[ b ] );
    }
    psSilk_VAD->counter = 15;

    /* init smoothed energy-to-noise ratio*/
    for( b = 0; b < VAD_N_BANDS; b++ ) {
        psSilk_VAD->NrgRatioSmth_Q8[ b ] = 100 * 256;       /* 100 * 256 --> 20 dB SNR */
    }

    return( ret );
}

/* Weighting factors for tilt measure */
static const opus_int32 tiltWeights[ VAD_N_BANDS ] = { 30000, 6000, -12000, -12000 };

/***************************************/
/* Get the speech activity level in Q8 */
/***************************************/
static opus_int silk_VAD_GetSA_Q8_c(                                   /* O    Return value, 0 if success                  */
    silk_encoder_state          *psEncC,                        /* I/O  Encoder state                               */
    const opus_int16            pIn[]                           /* I    PCM input                                   */
)
{
    opus_int   SA_Q15, pSNR_dB_Q7, input_tilt;
    opus_int   decimated_framelength1, decimated_framelength2;
    opus_int   decimated_framelength;
    opus_int   dec_subframe_length, dec_subframe_offset, SNR_Q7, i, b, s;
    opus_int32 sumSquared, smooth_coef_Q16;
    opus_int16 HPstateTmp;
    VARDECL( opus_int16, X );
    opus_int32 Xnrg[ VAD_N_BANDS ];
    opus_int32 NrgToNoiseRatio_Q8[ VAD_N_BANDS ];
    opus_int32 speech_nrg, x_tmp;
    opus_int   X_offset[ VAD_N_BANDS ];
    opus_int   ret = 0;
    silk_VAD_state *psSilk_VAD = &psEncC->sVAD;
    SAVE_STACK;

    /* Safety checks */
    silk_assert( VAD_N_BANDS == 4 );
    celt_assert( MAX_FRAME_LENGTH >= psEncC->frame_length );
    celt_assert( psEncC->frame_length <= 512 );
    celt_assert( psEncC->frame_length == 8 * silk_RSHIFT( psEncC->frame_length, 3 ) );

    /***********************/
    /* Filter and Decimate */
    /***********************/
    decimated_framelength1 = silk_RSHIFT( psEncC->frame_length, 1 );
    decimated_framelength2 = silk_RSHIFT( psEncC->frame_length, 2 );
    decimated_framelength = silk_RSHIFT( psEncC->frame_length, 3 );
    /* Decimate into 4 bands:
       0       L      3L       L              3L                             5L
               -      --       -              --                             --
               8       8       2               4                              4

       [0-1 kHz| temp. |1-2 kHz|    2-4 kHz    |            4-8 kHz           |

       They're arranged to allow the minimal ( frame_length / 4 ) extra
       scratch space during the downsampling process */
    X_offset[ 0 ] = 0;
    X_offset[ 1 ] = decimated_framelength + decimated_framelength2;
    X_offset[ 2 ] = X_offset[ 1 ] + decimated_framelength;
    X_offset[ 3 ] = X_offset[ 2 ] + decimated_framelength2;
    ALLOC( X, X_offset[ 3 ] + decimated_framelength1, opus_int16 );

    /* 0-8 kHz to 0-4 kHz and 4-8 kHz */
    silk_ana_filt_bank_1( pIn, &psSilk_VAD->AnaState[  0 ],
        X, &X[ X_offset[ 3 ] ], psEncC->frame_length );

    /* 0-4 kHz to 0-2 kHz and 2-4 kHz */
    silk_ana_filt_bank_1( X, &psSilk_VAD->AnaState1[ 0 ],
        X, &X[ X_offset[ 2 ] ], decimated_framelength1 );

    /* 0-2 kHz to 0-1 kHz and 1-2 kHz */
    silk_ana_filt_bank_1( X, &psSilk_VAD->AnaState2[ 0 ],
        X, &X[ X_offset[ 1 ] ], decimated_framelength2 );

    /*********************************************/
    /* HP filter on lowest band (differentiator) */
    /*********************************************/
    X[ decimated_framelength - 1 ] = silk_RSHIFT( X[ decimated_framelength - 1 ], 1 );
    HPstateTmp = X[ decimated_framelength - 1 ];
    for( i = decimated_framelength - 1; i > 0; i-- ) {
        X[ i - 1 ]  = silk_RSHIFT( X[ i - 1 ], 1 );
        X[ i ]     -= X[ i - 1 ];
    }
    X[ 0 ] -= psSilk_VAD->HPstate;
    psSilk_VAD->HPstate = HPstateTmp;

    /*************************************/
    /* Calculate the energy in each band */
    /*************************************/
    for( b = 0; b < VAD_N_BANDS; b++ ) {
        /* Find the decimated framelength in the non-uniformly divided bands */
        decimated_framelength = silk_RSHIFT( psEncC->frame_length, silk_min_int( VAD_N_BANDS - b, VAD_N_BANDS - 1 ) );

        /* Split length into subframe lengths */
        dec_subframe_length = silk_RSHIFT( decimated_framelength, VAD_INTERNAL_SUBFRAMES_LOG2 );
        dec_subframe_offset = 0;

        /* Compute energy per sub-frame */
        /* initialize with summed energy of last subframe */
        Xnrg[ b ] = psSilk_VAD->XnrgSubfr[ b ];
        for( s = 0; s < VAD_INTERNAL_SUBFRAMES; s++ ) {
            sumSquared = 0;
            for( i = 0; i < dec_subframe_length; i++ ) {
                /* The energy will be less than dec_subframe_length * ( silk_int16_MIN / 8 ) ^ 2.            */
                /* Therefore we can accumulate with no risk of overflow (unless dec_subframe_length > 128)  */
                x_tmp = silk_RSHIFT(
                    X[ X_offset[ b ] + i + dec_subframe_offset ], 3 );
                sumSquared = silk_SMLABB( sumSquared, x_tmp, x_tmp );

                /* Safety check */
                silk_assert( sumSquared >= 0 );
            }

            /* Add/saturate summed energy of current subframe */
            if( s < VAD_INTERNAL_SUBFRAMES - 1 ) {
                Xnrg[ b ] = silk_ADD_POS_SAT32( Xnrg[ b ], sumSquared );
            } else {
                /* Look-ahead subframe */
                Xnrg[ b ] = silk_ADD_POS_SAT32( Xnrg[ b ], silk_RSHIFT( sumSquared, 1 ) );
            }

            dec_subframe_offset += dec_subframe_length;
        }
        psSilk_VAD->XnrgSubfr[ b ] = sumSquared;
    }

    /********************/
    /* Noise estimation */
    /********************/
    silk_VAD_GetNoiseLevels( &Xnrg[ 0 ], psSilk_VAD );

    /***********************************************/
    /* Signal-plus-noise to noise ratio estimation */
    /***********************************************/
    sumSquared = 0;
    input_tilt = 0;
    for( b = 0; b < VAD_N_BANDS; b++ ) {
        speech_nrg = Xnrg[ b ] - psSilk_VAD->NL[ b ];
        if( speech_nrg > 0 ) {
            /* Divide, with sufficient resolution */
            if( ( Xnrg[ b ] & 0xFF800000 ) == 0 ) {
                NrgToNoiseRatio_Q8[ b ] = silk_DIV32( silk_LSHIFT( Xnrg[ b ], 8 ), psSilk_VAD->NL[ b ] + 1 );
            } else {
                NrgToNoiseRatio_Q8[ b ] = silk_DIV32( Xnrg[ b ], silk_RSHIFT( psSilk_VAD->NL[ b ], 8 ) + 1 );
            }

            /* Convert to log domain */
            SNR_Q7 = silk_lin2log( NrgToNoiseRatio_Q8[ b ] ) - 8 * 128;

            /* Sum-of-squares */
            sumSquared = silk_SMLABB( sumSquared, SNR_Q7, SNR_Q7 );          /* Q14 */

            /* Tilt measure */
            if( speech_nrg < ( (opus_int32)1 << 20 ) ) {
                /* Scale down SNR value for small subband speech energies */
                SNR_Q7 = silk_SMULWB( silk_LSHIFT( silk_SQRT_APPROX( speech_nrg ), 6 ), SNR_Q7 );
            }
            input_tilt = silk_SMLAWB( input_tilt, tiltWeights[ b ], SNR_Q7 );
        } else {
            NrgToNoiseRatio_Q8[ b ] = 256;
        }
    }

    /* Mean-of-squares */
    sumSquared = silk_DIV32_16( sumSquared, VAD_N_BANDS ); /* Q14 */

    /* Root-mean-square approximation, scale to dBs, and write to output pointer */
    pSNR_dB_Q7 = (opus_int16)( 3 * silk_SQRT_APPROX( sumSquared ) ); /* Q7 */

    /*********************************/
    /* Speech Probability Estimation */
    /*********************************/
    SA_Q15 = silk_sigm_Q15( silk_SMULWB( VAD_SNR_FACTOR_Q16, pSNR_dB_Q7 ) - VAD_NEGATIVE_OFFSET_Q5 );

    /**************************/
    /* Frequency Tilt Measure */
    /**************************/
    psEncC->input_tilt_Q15 = silk_LSHIFT( silk_sigm_Q15( input_tilt ) - 16384, 1 );

    /**************************************************/
    /* Scale the sigmoid output based on power levels */
    /**************************************************/
    speech_nrg = 0;
    for( b = 0; b < VAD_N_BANDS; b++ ) {
        /* Accumulate signal-without-noise energies, higher frequency bands have more weight */
        speech_nrg += ( b + 1 ) * silk_RSHIFT( Xnrg[ b ] - psSilk_VAD->NL[ b ], 4 );
    }

    if( psEncC->frame_length == 20 * psEncC->fs_kHz ) {
        speech_nrg = silk_RSHIFT32( speech_nrg, 1 );
    }
    /* Power scaling */
    if( speech_nrg <= 0 ) {
        SA_Q15 = silk_RSHIFT( SA_Q15, 1 );
    } else if( speech_nrg < 16384 ) {
        speech_nrg = silk_LSHIFT32( speech_nrg, 16 );

        /* square-root */
        speech_nrg = silk_SQRT_APPROX( speech_nrg );
        SA_Q15 = silk_SMULWB( 32768 + speech_nrg, SA_Q15 );
    }

    /* Copy the resulting speech activity in Q8 */
    psEncC->speech_activity_Q8 = silk_min_int( silk_RSHIFT( SA_Q15, 7 ), silk_uint8_MAX );

    /***********************************/
    /* Energy Level and SNR estimation */
    /***********************************/
    /* Smoothing coefficient */
    smooth_coef_Q16 = silk_SMULWB( VAD_SNR_SMOOTH_COEF_Q18, silk_SMULWB( (opus_int32)SA_Q15, SA_Q15 ) );

    if( psEncC->frame_length == 10 * psEncC->fs_kHz ) {
        smooth_coef_Q16 >>= 1;
    }

    for( b = 0; b < VAD_N_BANDS; b++ ) {
        /* compute smoothed energy-to-noise ratio per band */
        psSilk_VAD->NrgRatioSmth_Q8[ b ] = silk_SMLAWB( psSilk_VAD->NrgRatioSmth_Q8[ b ],
            NrgToNoiseRatio_Q8[ b ] - psSilk_VAD->NrgRatioSmth_Q8[ b ], smooth_coef_Q16 );

        /* signal to noise ratio in dB per band */
        SNR_Q7 = 3 * ( silk_lin2log( psSilk_VAD->NrgRatioSmth_Q8[b] ) - 8 * 128 );
        /* quality = sigmoid( 0.25 * ( SNR_dB - 16 ) ); */
        psEncC->input_quality_bands_Q15[ b ] = silk_sigm_Q15( silk_RSHIFT( SNR_Q7 - 16 * 128, 4 ) );
    }

    RESTORE_STACK;
    return( ret );
}

/**************************/
/* Noise level estimation */
/**************************/
# if  !defined(OPUS_X86_MAY_HAVE_SSE4_1)
static OPUS_INLINE
#endif
void silk_VAD_GetNoiseLevels(
    const opus_int32            pX[ VAD_N_BANDS ],  /* I    subband energies                            */
    silk_VAD_state              *psSilk_VAD         /* I/O  Pointer to Silk VAD state                   */
)
{
    opus_int   k;
    opus_int32 nl, nrg, inv_nrg;
    opus_int   coef, min_coef;

    /* Initially faster smoothing */
    if( psSilk_VAD->counter < 1000 ) { /* 1000 = 20 sec */
        min_coef = silk_DIV32_16( silk_int16_MAX, silk_RSHIFT( psSilk_VAD->counter, 4 ) + 1 );
        /* Increment frame counter */
        psSilk_VAD->counter++;
    } else {
        min_coef = 0;
    }

    for( k = 0; k < VAD_N_BANDS; k++ ) {
        /* Get old noise level estimate for current band */
        nl = psSilk_VAD->NL[ k ];
        silk_assert( nl >= 0 );

        /* Add bias */
        nrg = silk_ADD_POS_SAT32( pX[ k ], psSilk_VAD->NoiseLevelBias[ k ] );
        silk_assert( nrg > 0 );

        /* Invert energies */
        inv_nrg = silk_DIV32( silk_int32_MAX, nrg );
        silk_assert( inv_nrg >= 0 );

        /* Less update when subband energy is high */
        if( nrg > silk_LSHIFT( nl, 3 ) ) {
            coef = VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 >> 3;
        } else if( nrg < nl ) {
            coef = VAD_NOISE_LEVEL_SMOOTH_COEF_Q16;
        } else {
            coef = silk_SMULWB( silk_SMULWW( inv_nrg, nl ), VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 << 1 );
        }

        /* Initially faster smoothing */
        coef = silk_max_int( coef, min_coef );

        /* Smooth inverse energies */
        psSilk_VAD->inv_NL[ k ] = silk_SMLAWB( psSilk_VAD->inv_NL[ k ], inv_nrg - psSilk_VAD->inv_NL[ k ], coef );
        silk_assert( psSilk_VAD->inv_NL[ k ] >= 0 );

        /* Compute noise level by inverting again */
        nl = silk_DIV32( silk_int32_MAX, psSilk_VAD->inv_NL[ k ] );
        silk_assert( nl >= 0 );

        /* Limit noise levels (guarantee 7 bits of head room) */
        nl = silk_min( nl, 0x00FFFFFF );

        /* Store as part of state */
        psSilk_VAD->NL[ k ] = nl;
    }
}

/*=====================================================================
  Adapted from silk/float/find_pitch_lags_FLP.c and the parts of the
  encoder that drive it (silk/control_codec.c for the settings at
  complexity >= 8, silk/float/encode_frame_FLP.c for the VAD decision and
  the previous-frame state), and the silk_port_*() interface
  =====================================================================*/

float silk_port_burg( float * A, const float * x, int len, int order, float min_inv_gain )
{
    return silk_burg_modified_FLP( A, x, min_inv_gain, len, 1, order, 0 );
}

struct silk_port_vad {
    silk_encoder_state st;
};

silk_port_vad * silk_port_vad_create( int fs_kHz )
{
    silk_port_vad * vad = (silk_port_vad *)calloc( 1, sizeof( silk_port_vad ) );
    if( !vad ) return NULL;
    vad->st.fs_kHz = fs_kHz;
    vad->st.frame_length = SILK_PORT_FRAME_MS * fs_kHz;
    silk_VAD_Init( &vad->st.sVAD );
    return vad;
}

void silk_port_vad_destroy( silk_port_vad * vad )
{
    free( vad );
}

int silk_port_vad_frame( silk_port_vad * vad, const short * frame, int * input_tilt_Q15 )
{
    silk_VAD_GetSA_Q8_c( &vad->st, frame );
    if( input_tilt_Q15 ) *input_tilt_Q15 = vad->st.input_tilt_Q15;
    return vad->st.speech_activity_Q8;
}

struct silk_port_pitch {
    opus_int fs_kHz;
    opus_int frame_length;              /* 20 ms */
    opus_int ltp_mem_length;            /* 20 ms */
    opus_int la_pitch;                  /* 2 ms */
    opus_int pitch_LPC_win_length;      /* 24 ms */
    opus_int pitchEstimationLPCOrder;
    opus_int pitchEstimationComplexity;
    opus_int32 pitchEstimationThreshold_Q16;
    opus_int prevLag;
    opus_int prevSignalType;
    silk_float LTPCorr;                 /* normalized correlation at the last lag; it
                                           biases the next search toward that lag */
    opus_int first_frame_after_reset;
    opus_int vad_gating;                /* skip the search when the VAD finds no
                                           activity, as SILK does (default: off) */
};

silk_port_pitch * silk_port_pitch_create( int fs_kHz )
{
    silk_port_pitch * p = (silk_port_pitch *)calloc( 1, sizeof( silk_port_pitch ) );
    if( !p ) return NULL;
    p->fs_kHz = fs_kHz;
    p->frame_length = SILK_PORT_FRAME_MS * fs_kHz;
    p->ltp_mem_length = LTP_MEM_LENGTH_MS * fs_kHz;
    p->la_pitch = LA_PITCH_MS * fs_kHz;
    p->pitch_LPC_win_length = FIND_PITCH_LPC_WIN_MS * fs_kHz;
    /* silk_setup_complexity(), complexity >= 8 */
    p->pitchEstimationComplexity = SILK_PE_MAX_COMPLEX;
    p->pitchEstimationThreshold_Q16 = SILK_FIX_CONST( 0.7, 16 );
    p->pitchEstimationLPCOrder = 16;
    p->first_frame_after_reset = 1;
    p->vad_gating = 0;
    return p;
}

void silk_port_pitch_set_vad_gating( silk_port_pitch * p, int on )
{
    p->vad_gating = on != 0;
}

void silk_port_pitch_destroy( silk_port_pitch * p )
{
    free( p );
}

int silk_port_pitch_buffer_length( int fs_kHz )
{
    return ( LTP_MEM_LENGTH_MS + SILK_PORT_FRAME_MS + LA_PITCH_MS ) * fs_kHz;
}

int silk_port_pitch_frame( silk_port_pitch * p, const float * x_buf, int speech_activity_Q8,
                           int input_tilt_Q15, int lags[ SILK_PORT_NB_SUBFR ] )
{
    opus_int   buf_len, signalType;
    opus_int16 lagIndex;
    opus_int8  contourIndex;
    silk_float thrhld, res_nrg;
    const silk_float *x_buf_ptr;
    silk_float auto_corr[ MAX_FIND_PITCH_LPC_ORDER + 1 ];
    silk_float A[         MAX_FIND_PITCH_LPC_ORDER ];
    silk_float refl_coef[ MAX_FIND_PITCH_LPC_ORDER ];
    silk_float Wsig[      FIND_PITCH_LPC_WIN_MAX ];
    silk_float res[       ( LTP_MEM_LENGTH_MS + MAX_FRAME_LENGTH_MS + LA_PITCH_MS ) * MAX_FS_KHZ ];
    silk_float *Wsig_ptr;

    /* silk_encode_do_VAD_FLP(): the signal type from the speech activity */
    signalType = speech_activity_Q8 < SILK_FIX_CONST( SPEECH_ACTIVITY_DTX_THRES, 8 )
                 ? TYPE_NO_VOICE_ACTIVITY : TYPE_UNVOICED;

    /******************************************/
    /* Set up buffer lengths etc based on Fs  */
    /******************************************/
    buf_len = p->la_pitch + p->frame_length + p->ltp_mem_length;

    /******************************************/
    /* Estimate LPC AR coefficients           */
    /******************************************/

    /* Calculate windowed signal */

    /* First LA_LTP samples */
    x_buf_ptr = x_buf + buf_len - p->pitch_LPC_win_length;
    Wsig_ptr  = Wsig;
    silk_apply_sine_window_FLP( Wsig_ptr, x_buf_ptr, 1, p->la_pitch );

    /* Middle non-windowed samples */
    Wsig_ptr  += p->la_pitch;
    x_buf_ptr += p->la_pitch;
    silk_memcpy( Wsig_ptr, x_buf_ptr, ( p->pitch_LPC_win_length - ( p->la_pitch << 1 ) ) * sizeof( silk_float ) );

    /* Last LA_LTP samples */
    Wsig_ptr  += p->pitch_LPC_win_length - ( p->la_pitch << 1 );
    x_buf_ptr += p->pitch_LPC_win_length - ( p->la_pitch << 1 );
    silk_apply_sine_window_FLP( Wsig_ptr, x_buf_ptr, 2, p->la_pitch );

    /* Calculate autocorrelation sequence */
    silk_autocorrelation_FLP( auto_corr, Wsig, p->pitch_LPC_win_length, p->pitchEstimationLPCOrder + 1, 0 );

    /* Add white noise, as a fraction of the energy */
    auto_corr[ 0 ] += auto_corr[ 0 ] * FIND_PITCH_WHITE_NOISE_FRACTION + 1;

    /* Calculate the reflection coefficients using Schur */
    res_nrg = silk_schur_FLP( refl_coef, auto_corr, p->pitchEstimationLPCOrder );
    (void)res_nrg;

    /* Convert reflection coefficients to prediction coefficients */
    silk_k2a_FLP( A, refl_coef, p->pitchEstimationLPCOrder );

    /* Bandwidth expansion */
    silk_bwexpander_FLP( A, p->pitchEstimationLPCOrder, FIND_PITCH_BANDWIDTH_EXPANSION );

    /*****************************************/
    /* LPC analysis filtering                */
    /*****************************************/
    silk_LPC_analysis_filter_FLP( res, A, x_buf, buf_len, p->pitchEstimationLPCOrder );

    /* SILK only searches frames with voice activity. Without VAD gating
       (rt_lpc's default) every frame is searched: the VAD's noise estimate
       can take a steady vowel for noise, especially in the first seconds,
       and an unsearched frame would come out unvoiced. Low activity still
       raises the threshold below. */
    if( ( signalType != TYPE_NO_VOICE_ACTIVITY || !p->vad_gating ) && p->first_frame_after_reset == 0 ) {
        /* Threshold for pitch estimator */
        thrhld  = 0.6f;
        thrhld -= 0.004f * p->pitchEstimationLPCOrder;
        thrhld -= 0.1f   * speech_activity_Q8 * ( 1.0f /  256.0f );
        thrhld -= 0.15f  * (p->prevSignalType >> 1);
        thrhld -= 0.1f   * input_tilt_Q15 * ( 1.0f / 32768.0f );

        /*****************************************/
        /* Call Pitch estimator                  */
        /*****************************************/
        if( silk_pitch_analysis_core_FLP( res, lags, &lagIndex,
            &contourIndex, &p->LTPCorr, p->prevLag, p->pitchEstimationThreshold_Q16 / 65536.0f,
            thrhld, p->fs_kHz, p->pitchEstimationComplexity, SILK_PORT_NB_SUBFR, 0 ) == 0 )
        {
            signalType = TYPE_VOICED;
        } else {
            signalType = TYPE_UNVOICED;
        }
    } else {
        silk_memset( lags, 0, SILK_PORT_NB_SUBFR * sizeof( int ) );
        p->LTPCorr = 0;
    }

    /* silk_encode_frame_FLP(): parameters needed for the next frame */
    p->prevLag = lags[ SILK_PORT_NB_SUBFR - 1 ];
    p->prevSignalType = signalType;
    p->first_frame_after_reset = 0;
    return signalType;
}
