//-----------------------------------------------------------------------------
// name: lpc.h
// desc: linear predictive coding: analysis, synthesis, reflection
//       coefficients, and formant measurement and shifting (lpc.cpp)
//
// Conventions:
//   - coefs[0..order-1] are predictor coefficients a1..ap:
//         prediction of x[n] = coefs[0] x[n-1] + ... + coefs[order-1] x[n-order]
//     so the analysis filter is A(z) = 1 - sum a_k z^-k and the synthesis
//     (vocal tract) filter is 1 / A(z).
//   - "pitch" is a pitch PERIOD in samples, not a frequency; 0 = unvoiced.
//   - Functions that take an lpc_data keep their state in it; use one
//     instance per audio stream. The formant functions use static work
//     buffers and aren't thread-safe.
//-----------------------------------------------------------------------------
#ifndef __LPC_H__
#define __LPC_H__

typedef float SAMPLE;

// one LPC analyzer/synthesizer (opaque; see lpc.cpp)
struct lpc_data_;
typedef lpc_data_ * lpc_data;




//-----------------------------------------------------------------------------
// instances
//-----------------------------------------------------------------------------
// a new analyzer/synthesizer, with the built-in glottal pulse installed
lpc_data lpc_create( );
// frees it and sets the pointer to NULL
void lpc_destroy( lpc_data & lpc );
// the pitch range the analysis looks for, in Hz, at sample rate srate
// (default: periods of 2% .. 40% of the frame, about 60 .. 1200 Hz for
// 1024-sample frames at 24 kHz). Periods longer than half a frame are never
// detected.
void lpc_set_pitch_range( lpc_data lpc, float srate, float lowest_hz, float highest_hz );
// installs a copy of alt[0..len-1] as the glottal pulse for voiced
// synthesis (NULL or 0 removes it: impulse train instead)
void lpc_alt( lpc_data lpc, SAMPLE * alt, int len );




//-----------------------------------------------------------------------------
// analysis
//-----------------------------------------------------------------------------
// LPC analysis of one frame x[0..len-1]: coefs[0..order-1], a gain (*power,
// the excitation level lpc_excitation() expects), the pitch period (*pitch,
// 0 if unvoiced) and, if residue isn't NULL, the prediction error
// residue[0..len-1]. The coefficients are fitted to the frame under a Hann
// window, with the Levinson-Durbin recursion (always a stable filter).
void lpc_analyze( lpc_data lpc, SAMPLE * x, int len, float * coefs, int order,
                  float * power, float * pitch, SAMPLE * residue );
// runs the prediction-error filter over x[0..len-1] (residue may be NULL);
// returns the gain lpc_analyze() reports
float lpc_predict( lpc_data lpc, SAMPLE * x, int len, float * coefs, int order,
                   SAMPLE * residue );
// autocorrelation of x[0..len-1] into y[0..len-1], and the pitch period
// found in it between min_lag and max_lag samples (0 = defaults), or 0 if
// the frame isn't clearly periodic
float autocorrelate_range( SAMPLE * x, int len, SAMPLE * y, float min_lag, float max_lag );
// first-order pre-emphasis x[i] -= alpha x[i-1], and the 2-tap
// de-emphasis x[i] = alpha x[i] + (1-alpha) x[i-1], both in place
void lpc_preemphasis( SAMPLE * x, int len, float alpha );
void lpc_deemphasis( SAMPLE * x, int len, float alpha );




//-----------------------------------------------------------------------------
// synthesis
//-----------------------------------------------------------------------------
// len samples of excitation: noise if pitch is 0, otherwise one pulse per
// period (the glottal pulse if alt is non-zero and one is installed, else
// an impulse), scaled by power. The pulse timing carries on across calls.
void lpc_excitation( lpc_data lpc, SAMPLE * e, int len, float power, float pitch, int alt );
// e[0..len-1] through 1/A(z) into y (which may be e). state[0..order-1] is
// the filter's history (its last outputs, newest first), read and updated.
void lpc_filter( const SAMPLE * e, SAMPLE * y, int len, const float * coefs, int order, float * state );
// lpc_excitation() through 1/A(z), with the filter history kept in lpc
void lpc_synthesize( lpc_data lpc, SAMPLE * y, int len, float * coefs,
                     int order, float power, float pitch, int alt );




//-----------------------------------------------------------------------------
// reflection (PARCOR) coefficients
//   1/A(z) is stable exactly when every reflection coefficient is strictly
//   between -1 and 1, and any weighted average of two stable sets is
//   stable: two filters can be blended smoothly by interpolating their
//   reflection coefficients. Zero-padding a set to a higher order changes
//   nothing.
//-----------------------------------------------------------------------------
#define LPC_REFLECTION_MAX_ORDER 256
// returns false if coefs isn't a stable filter; the offending reflection
// coefficients are then clamped to +/-0.999 (non-numbers set to 0), which
// gives a stable filter close to it. order <= LPC_REFLECTION_MAX_ORDER.
bool lpc_to_reflection( const float * coefs, int order, float * k );
void reflection_to_lpc( const float * k, int order, float * coefs );




//-----------------------------------------------------------------------------
// formants
//   Three methods, each with a measuring function and a shifting ("warp")
//   function:
//     envelope  peaks of the sampled spectral envelope; the warp stretches
//               the envelope along the frequency axis and refits it
//     poles     the conjugate pole pairs of 1/A(z); the warp moves the
//               formant poles only (bandwidths scaled by bw_factor: 1 keeps
//               them in Hz, alpha scales them with the frequencies)
//     lsf       close pairs of line spectral frequencies; the warp moves
//               the LSFs (the safest representation to edit)
//   The warps multiply every formant frequency by alpha (> 1 raises them,
//   like a shorter vocal tract), keep the loudness the same by adjusting
//   *power, and leave coefs and *power alone if anything fails
//   numerically. before / after (optional) receive the formants of the
//   original filter and of the result. srate is the sample rate the
//   coefficients are for. order <= 1023.
//-----------------------------------------------------------------------------
#define LPC_MAX_FORMANTS 5          // most formants reported per filter

struct Formant
{
    double freq;        // center frequency in Hz
    double bw;          // -3 dB bandwidth in Hz
    double level;       // envelope level at the peak, in dB
};

struct FormantSet
{
    int count;                          // how many of f[] are valid
    Formant f[LPC_MAX_FORMANTS];        // in increasing frequency: f[0] = F1, ...
};

void lpc_formants( const float * coefs, int order, float srate, FormantSet * out );
void warp_formants( float * coefs, int order, float srate, float alpha, float * power,
                    FormantSet * before = 0, FormantSet * after = 0 );

void lpc_formants_poles( const float * coefs, int order, float srate, FormantSet * out );
void warp_formants_poles( float * coefs, int order, float srate, float alpha, float * power,
                          FormantSet * before = 0, FormantSet * after = 0,
                          float bw_factor = 1.0f );

void lpc_formants_lsf( const float * coefs, int order, float srate, FormantSet * out );
void warp_formants_lsf( float * coefs, int order, float srate, float alpha, float * power,
                        FormantSet * before = 0, FormantSet * after = 0 );

// measures the resonances of a frame of speech x[0..n-1] (n <= 1024) at
// sample rate fs with a standard formant-tracking analysis (order-14 LPC at
// about 12 kHz): up to max [frequency, bandwidth] pairs in Hz, lowest
// first. Returns how many.
int lpc_measure_formants( const SAMPLE * x, int n, double fs, double * freq, double * bw, int max );

#endif
