/*-----------------------------------------------------------------------------
 silk_port.h -- the analysis parts of the SILK speech codec, ported from
 libopus (see silk_port.c for what's included and the license), with a
 small interface for rt_lpc.

 Signal scale: like SILK's float code, these work on samples scaled as
 16-bit integers (+/-32768), not +/-1.
-----------------------------------------------------------------------------*/
#ifndef SILK_PORT_H
#define SILK_PORT_H

#ifdef __cplusplus
extern "C" {
#endif

/* SILK frames are 20 ms: 4 subframes of 5 ms. Its pitch analysis looks at
   20 ms of history, the 20 ms frame, and 2 ms of lookahead. */
#define SILK_PORT_NB_SUBFR      4
#define SILK_PORT_FRAME_MS      20
#define SILK_PORT_LTP_MEM_MS    20
#define SILK_PORT_LA_PITCH_MS   2

/* signal types */
#define SILK_PORT_NO_VOICE_ACTIVITY  0
#define SILK_PORT_UNVOICED           1
#define SILK_PORT_VOICED             2

/* SILK limits the LPC fit's prediction gain to this (40 dB) */
#define SILK_PORT_MAX_PREDICTION_POWER_GAIN  1e4f

/* LPC coefficients A[0..order-1] (predictor convention: prediction =
   sum A[k] x[n-1-k]) of x[0..len-1] by Burg's method, SILK's
   silk_burg_modified_FLP(). The first `order` samples serve only as
   history. The prediction gain is capped at 1 / min_inv_gain. Returns the
   residual energy. order <= 128. */
float silk_port_burg( float * A, const float * x, int len, int order, float min_inv_gain );

/* SILK's voice activity detector (silk_VAD_GetSA_Q8()), at fs_kHz = 8, 12
   or 16, on 20 ms frames */
typedef struct silk_port_vad silk_port_vad;
silk_port_vad * silk_port_vad_create( int fs_kHz );
void silk_port_vad_destroy( silk_port_vad * vad );
/* one frame (20 ms: 20 * fs_kHz samples). Returns the speech activity,
   0 .. 255 (Q8); *input_tilt_Q15 gets the spectral tilt measure. */
int silk_port_vad_frame( silk_port_vad * vad, const short * frame, int * input_tilt_Q15 );

/* SILK's pitch estimator (silk_find_pitch_lags_FLP() and
   silk_pitch_analysis_core_FLP() at their highest complexity), at fs_kHz =
   8, 12 or 16. Keeps the previous frame's lag and signal type, which bias
   the next search, as the encoder does. */
typedef struct silk_port_pitch silk_port_pitch;
silk_port_pitch * silk_port_pitch_create( int fs_kHz );
void silk_port_pitch_destroy( silk_port_pitch * p );
/* SILK doesn't search for pitch in frames whose speech activity is below
   its threshold (5%): they're "no voice activity". That suits a codec,
   which spends no bits on them, but the VAD's noise estimate can take a
   steady vowel for noise (quickly so in the first seconds, when it adapts
   fast), so for voice transformation, where an unsearched voiced frame
   would lose its pitch, the default here is to search every frame. On (1)
   gives SILK's behavior exactly. */
void silk_port_pitch_set_vad_gating( silk_port_pitch * p, int on );
/* samples silk_port_pitch_frame() needs: (20 + 20 + 2) ms at fs_kHz */
int silk_port_pitch_buffer_length( int fs_kHz );
/* analyzes one frame. x holds silk_port_pitch_buffer_length() samples:
   20 ms of history, the 20 ms frame, 2 ms of lookahead. speech_activity_Q8
   and input_tilt_Q15 come from the VAD for this frame; below SILK's
   activity threshold the frame is "no voice activity" (and with VAD gating
   on, isn't searched).
   Writes the pitch lag (samples at fs_kHz) of each of the frame's four
   subframes to lags[] (zeros unless voiced), and returns the signal type
   (SILK_PORT_*). */
int silk_port_pitch_frame( silk_port_pitch * p, const float * x, int speech_activity_Q8,
                           int input_tilt_Q15, int lags[ SILK_PORT_NB_SUBFR ] );

#ifdef __cplusplus
}
#endif

#endif
