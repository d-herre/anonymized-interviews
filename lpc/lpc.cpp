//-----------------------------------------------------------------------------
// name: lpc.cpp
// desc: linear predictive coding
//
// authors: Ananya Misra (amisra@cs.princeton.edu)
//          Ge Wang (gewang@cs.princeton.edu)
//
// based (heavily) on filtlpc.c and lpcresyn.c by Perry Cook
//
// Also here: conversions between LPC coefficients and reflection
// coefficients, and formant measurement and formant shifting (moved from
// rt_lpc.cpp).
//
// date: today
//-----------------------------------------------------------------------------
#include "lpc.h"
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <limits.h>

#define LPC__PI 3.14159265358979323846




//-----------------------------------------------------------------------------
// Overview
//
// LPC (linear predictive coding) treats a short frame of speech as the
// output of an all-pole filter (the vocal tract) driven by a simple
// excitation signal (the vocal folds, or noise):
//
//     x[n] = a1*x[n-1] + a2*x[n-2] + ... + ap*x[n-p] + e[n]
//
// Each sample is "predicted" from the p samples before it; e[n] is what the
// prediction misses (the "residue").
//
// - Analysis (lpc_analyze) estimates a1..ap (stored as coefs[0..p-1]) using
//   the autocorrelation method: from the (windowed) frame's autocorrelation
//   r(), solve the normal equations
//       sum_j a_j r(|i-j|) = r(i),  i = 1..p
//   with the Levinson-Durbin recursion. It also measures the residue, a
//   gain ("power"), and a pitch estimate.
//
// - Synthesis (lpc_synthesize) runs the other direction: generate an
//   excitation (noise, an impulse train, or a stored glottal pulse) and pass
//   it through the all-pole filter built from the coefficients.
//
// NOTE: throughout this file (and rt_lpc.cpp) "pitch" is a pitch PERIOD in
// samples, not a frequency in Hz. A pitch of 0 means "unvoiced".
//-----------------------------------------------------------------------------





// internal state of one LPC analyzer/synthesizer (lpc_data is a pointer to this)
struct lpc_data_
{
    SAMPLE * corr;      // autocorrelation of the current frame (len values)
    SAMPLE * Zs;        // analysis filter history: the last `order` INPUT
                        //   samples, newest first (Zs[0] = most recent)
    SAMPLE * Zss;       // synthesis filter history: the last `order` OUTPUT
                        //   samples, newest first. Kept between calls so the
                        //   filter runs continuously across buffers.
    SAMPLE * alt;       // alternate excitation waveform used for voiced
                        //   synthesis (default: the glottal pulse below)
    int order;          // order that fit/Zs/Zss are currently sized for
    int len;            // frame length that corr is currently sized for
    int alt_len;        // number of samples in alt
    int ticker;         // synthesis: samples left until the next pitch pulse.
                        //   Kept between calls so the pulse train stays
                        //   continuous across buffers.
    int which;          // unused
    float min_lag;      // pitch search range: shortest and longest period in
    float max_lag;      //   samples (lpc_set_pitch_range()); 0 = defaults
    float * win;        // Hann window for the LPC fit (len values)
    double * fit;       // LPC fit scratch: windowed autocorrelation and
                        //   Levinson-Durbin work space (3 * (order+1) values)
};

// One recorded glottal pulse (the puff of air released by the vocal folds
// each pitch period), originally the file glot_pop.raw. Values are 16-bit
// sample values, which is why they are divided by SHRT_MAX where used.
// lpc_create() installs this as the default "alt" excitation, which
// lpc_synthesize() plays once per pitch period instead of a bare impulse.
static SAMPLE glot_pop_data[] = {
    0.0f,0.0f,1.0f,-1.0f,0.0f,0.0f,0.0f,0.0f,
    1.0f,1.0f,513.0f,561.0f,565.0f,499.0f,374.0f,188.0f,
    -42.0f,-306.0f,-571.0f,-821.0f,-1037.0f,-1214.0f,-1338.0f,-1422.0f,
    -1479.0f,-1520.0f,-1568.0f,-1628.0f,-1708.0f,-1814.0f,-1931.0f,-2062.0f,
    -2185.0f,-2293.0f,-2375.0f,-2432.0f,-2464.0f,-2476.0f,-2471.0f,-2458.0f,
    -2431.0f,-2397.0f,-2338.0f,-2247.0f,-2110.0f,-1916.0f,-1655.0f,-1330.0f,
    -950.0f,-528.0f,-91.0f,337.0f,730.0f,1059.0f,1310.0f,1467.0f,
    1531.0f,1509.0f,1411.0f,1256.0f,1066.0f,847.0f,629.0f,415.0f,
    217.0f,37.0f,-131.0f,-278.0f,-411.0f,-524.0f,-624.0f,-711.0f,
    -778.0f,-827.0f,-859.0f,-867.0f,-866.0f,-846.0f,-824.0f,-804.0f,
    -788.0f,-789.0f,-808.0f,-855.0f,-926.0f,-1019.0f,-1142.0f,-1282.0f,
    -1441.0f,-1617.0f,-1803.0f,-2007.0f,-2223.0f,-2446.0f,-2670.0f,-2886.0f,
    -3077.0f,-3235.0f,-3330.0f,-3347.0f,-3278.0f,-3107.0f,-2844.0f,-2499.0f,
    -2102.0f,-1674.0f,-1253.0f,-869.0f,-548.0f,-299.0f,-131.0f,-29.0f,
    24.0f,59.0f,94.0f,150.0f,240.0f,368.0f,519.0f,688.0f,
    856.0f,994.0f,1093.0f,1144.0f,1135.0f,1077.0f,968.0f,823.0f,
    646.0f,458.0f,268.0f,82.0f,-93.0f,-255.0f,-395.0f,-523.0f,
    -641.0f,-749.0f,-858.0f,-979.0f,-1114.0f,-1266.0f,-1437.0f,-1610.0f,
    -1779.0f,-1925.0f,-2028.0f,-2073.0f,-2047.0f,-1953.0f,-1795.0f,-1592.0f,
    -1374.0f,-1167.0f,-995.0f,-888.0f,-851.0f,-891.0f,-983.0f,-1112.0f,
    -1239.0f,-1325.0f,-1345.0f,-1266.0f,-1081.0f,-788.0f,-412.0f,21.0f,
    465.0f,881.0f,1223.0f,1460.0f,1576.0f,1556.0f,1423.0f,1201.0f,
    934.0f,660.0f,430.0f,280.0f,234.0f,300.0f,459.0f,685.0f,
    930.0f,1143.0f,1275.0f,1285.0f,1145.0f,866.0f,465.0f,-8.0f,
    -490.0f,-913.0f,-1208.0f,-1322.0f,-1233.0f,-941.0f,-472.0f,96.0f,
    701.0f,1240.0f,1641.0f,1848.0f,1830.0f,1590.0f,1181.0f,678.0f,
    174.0f,-238.0f,-481.0f,-502.0f,-293.0f,120.0f,687.0f,1322.0f,
    1943.0f,2470.0f,2853.0f,3075.0f,3151.0f,3120.0f,3036.0f,2957.0f,
    2929.0f,2977.0f,3082.0f,3206.0f,3299.0f,3295.0f,3153.0f,2863.0f,
    2431.0f,1916.0f,1405.0f,999.0f,799.0f,891.0f,1299.0f,2024.0f,
    2997.0f,4118.0f,5278.0f,6312.0f,7105.0f,7554.0f,7625.0f,7328.0f,
    6724.0f,5925.0f,5063.0f,4280.0f,3700.0f,3391.0f,3411.0f,3737.0f,
    4307.0f,5027.0f,5773.0f,6436.0f,6918.0f,7164.0f,7178.0f,7003.0f,
    6733.0f,6509.0f,6468.0f,6734.0f,7386.0f,8444.0f,9841.0f,11430.0f,
    12991.0f,14273.0f,15014.0f,14983.0f,14047.0f,12167.0f,9434.0f,6071.0f,
    2379.0f,-1263.0f,-4504.0f,-7035.0f,-8657.0f,-9297.0f,-9036.0f,-8060.0f,
    -6653.0f,-5126.0f,-3761.0f,-2782.0f,-2285.0f,-2276.0f,-2655.0f,-3251.0f,
    -3867.0f,-4324.0f,-4512.0f,-4379.0f,-3970.0f,-3372.0f,-2719.0f,-2129.0f,
    -1704.0f,-1489.0f,-1470.0f,-1589.0f,-1747.0f,-1853.0f,-1844.0f,-1693.0f,
    -1423.0f,-1108.0f,-819.0f,-663.0f,-707.0f,-966.0f,-1411.0f,-1953.0f,
    -2478.0f,-2836.0f,-2913.0f,-2625.0f,-1947.0f,-922.0f,362.0f,1760.0f,
    3116.0f,4278.0f,5127.0f,5586.0f,5646.0f,5340.0f,4750.0f,3974.0f,
    3115.0f,2269.0f,1495.0f,822.0f,254.0f,-237.0f,-678.0f,-1090.0f,
    -1487.0f,-1856.0f,-2172.0f,-2406.0f,-2515.0f,-2473.0f,-2289.0f,-1978.0f,
    -1593.0f,-1201.0f,-876.0f,-681.0f,-661.0f,-836.0f,-1184.0f,-1664.0f,
    -2198.0f,-2717.0f,-3148.0f,-3432.0f,-3544.0f,-3495.0f,-3322.0f,-3073.0f,
    -2829.0f,-2648.0f,-2581.0f,-2666.0f,-2885.0f,-3213.0f,-3593.0f,-3957.0f,
    -4233.0f,-4365.0f,-4312.0f,-4064.0f,-3636.0f,-3057.0f,-2378.0f,-1656.0f,
    -945.0f,-289.0f,262.0f,697.0f,1002.0f,1167.0f,1205.0f,1115.0f,
    917.0f,630.0f,272.0f,-130.0f,-548.0f,-946.0f,-1294.0f,-1572.0f,
    -1753.0f,-1837.0f,-1835.0f,-1763.0f,-1642.0f,-1510.0f,-1391.0f,-1303.0f,
    -1259.0f,-1251.0f,-1264.0f,-1281.0f,-1271.0f,-1229.0f,-1139.0f,-1000.0f,
    -835.0f,-669.0f,-530.0f,-446.0f,-440.0f,-517.0f,-678.0f,-905.0f,
    -1181.0f,-1471.0f,-1750.0f,-1999.0f,-2197.0f,-2333.0f,-2411.0f,-2422.0f,
    -2375.0f,-2259.0f,-2090.0f,-1850.0f,-1543.0f,-1169.0f,-737.0f,-260.0f,
    232.0f,707.0f,1122.0f,1444.0f,1639.0f,1688.0f,1586.0f,1347.0f,
    1010.0f,615.0f,212.0f,-153.0f,-440.0f,-635.0f,-733.0f,-759.0f,
    -745.0f,-744.0f,-793.0f,-930.0f,-1170.0f,-1513.0f,-1938.0f,-2401.0f,
    -2860.0f,-3257.0f,-3553.0f,-3720.0f,-3747.0f,-3643.0f,-3430.0f,-3141.0f,
    -2811.0f,-2480.0f,-2170.0f,-1903.0f,-1676.0f,-1497.0f,-1356.0f,-1241.0f,
    -1151.0f,-1083.0f,-1041.0f,-1016.0f,-1017.0f,-1041.0f,-1081.0f,-1127.0f,
    -1165.0f,-1182.0f,-1164.0f,-1110.0f,-1007.0f,-871.0f,-706.0f,-533.0f,
    -361.0f,-211.0f,-89.0f,-4.0f,51.0f,68.0f,63.0f,48.0f,
    24.0f,5.0f,-12.0f,-22.0f,-26.0f,-23.0f,-17.0f,-10.0f,
    -5.0f,2.0f,4.0f,6.0f,6.0f,6.0f,5.0f,0.0f,
    0.0f,
    0
}; static int glot_pop_size = 521;




//-----------------------------------------------------------------------------
// name: lpc_create()
// desc: allocates a new LPC analyzer/synthesizer with zeroed state and the
//       default glottal pulse as its alternate excitation. The working
//       buffers are sized later, on the first call to lpc_analyze().
//-----------------------------------------------------------------------------
lpc_data lpc_create( )
{
    lpc_data instance = new lpc_data_;
    memset( instance, 0, sizeof(lpc_data_) );
    // set the default glottal pulse
    lpc_alt( instance, glot_pop_data, glot_pop_size );

    return instance;
}




//-----------------------------------------------------------------------------
// name: lpc_set_pitch_range()
// desc: sets the range of pitches the analysis looks for, in Hz, at sample
//       rate srate. Below lowest_hz a frame is treated as unvoiced. The
//       longest period that can be detected is also capped at half the
//       analysis frame (so at least one whole period always overlaps), so a
//       frame of N samples can't go below 2 * srate / N Hz however low
//       lowest_hz is.
//       Without this, the defaults are periods from 2% to 40% of the frame
//       length: about 60 .. 1200 Hz for 1024-sample frames at 24 kHz.
//-----------------------------------------------------------------------------
void lpc_set_pitch_range( lpc_data lpc, float srate, float lowest_hz, float highest_hz )
{
    if( !lpc || !( srate > 0 && lowest_hz > 0 && highest_hz > lowest_hz ) ) return;
    lpc->min_lag = srate / highest_hz;
    lpc->max_lag = srate / lowest_hz;
}




//-----------------------------------------------------------------------------
// name: lpc_destroy()
// desc: frees an LPC instance and sets the caller's pointer to NULL
//-----------------------------------------------------------------------------
void lpc_destroy( lpc_data & instance )
{
    if( instance )
    {
        if( instance->corr )
            delete [] instance->corr;
        if( instance->Zs )
            delete [] instance->Zs;
        if( instance->Zss )
            delete [] instance->Zss;
        if( instance->alt )
            delete [] instance->alt;
        if( instance->win )
            delete [] instance->win;
        if( instance->fit )
            delete [] instance->fit;

        delete instance;
        instance = NULL;
    }
}




//-----------------------------------------------------------------------------
// name: lpc_predict()
// desc: runs the LPC analysis ("prediction error") filter over x[0..len-1].
//       Each sample x[i] is predicted from the `order` samples before it:
//           prediction = coefs[0]*x[i-1] + coefs[1]*x[i-2] + ...
//       and the error x[i] - prediction is written to residue[i] (if
//       residue is not NULL). The first `order` samples don't have a full
//       history, so their residue is left at 0.
//       Returns a gain estimate: sqrt( sum of squared errors ) / (len - order).
//       (Not quite the RMS error, which would be sqrt( sum / (len - order) );
//       the synthesizer's scaling constants are tuned to this value.)
//-----------------------------------------------------------------------------
float lpc_predict( lpc_data lpc, SAMPLE * x, int len, float * coefs, int order, 
                   SAMPLE * residue )
{
    int i,j;
    float power = 0.0f;
    float error, tmp;
    // seed the history with the first `order` samples, newest first:
    //   Zs[0] = x[order-1], Zs[1] = x[order-2], ..., Zs[order-1] = x[0]
    for( i = 0; i < order; i++ )
        lpc->Zs[i] = x[order - i - 1];

    // number of samples that have a full history and can be predicted
    // ("hope" is presumably a typo for "hop")
    int hope_size = len - order;
    
    // zero out the residue
    if( residue ) memset( residue, 0, len * sizeof(SAMPLE) );

    // predict each sample and accumulate the squared prediction error
    for( i = order; i < hope_size + order; i++ )
    {
        // prediction of x[i] from the previous `order` samples
        tmp = 0.0f;
        for( j = 0; j < order; j++ ) tmp += lpc->Zs[j] * coefs[j];
        // shift the history by one (dropping the oldest) and add x[i]
        for( j = order - 1; j > 0; j-- ) lpc->Zs[j] = lpc->Zs[j-1];
        lpc->Zs[0] = x[i];
        error = x[i] - tmp;
        power += error * error;
        
        if( residue ) residue[i] = error;
    }

    // gain estimate (see desc above)
    return (float)sqrt(power) / hope_size;
}




//-----------------------------------------------------------------------------
// name: lpc_analyze()
// desc: LPC analysis of one frame x[0..len-1].
//       outputs: coefs[0..order-1]  predictor coefficients a1..ap
//                *power             excitation gain (from lpc_predict)
//                *pitch             pitch period in samples, 0 = unvoiced
//                residue[0..len-1]  prediction error signal (may be NULL)
//       Internal buffers are reallocated whenever len or order changes.
//-----------------------------------------------------------------------------
void lpc_analyze( lpc_data lpc, SAMPLE * x, int len, float * coefs, int order,
                  float * power, float * pitch, SAMPLE * residue )
{
    int i, j;

    // (re)size the autocorrelation buffer if the frame length changed
    if( lpc->len != len )
    {
        if( lpc->corr ) delete [] lpc->corr;
        if( lpc->win ) delete [] lpc->win;
        lpc->corr = new SAMPLE[len];
        lpc->win = new float[len];
        for( i = 0; i < len; i++ )
            lpc->win[i] = (float)( .5 - .5 * cos( 2.0 * M_PI * ( i + .5 ) / len ) );
        lpc->len = len;
    }
    
    // (re)size the fit buffers and filter histories if the order changed.
    // This also resets the synthesis filter state and pulse timing.
    if( lpc->order != order )
    {
        if( lpc->Zs ) delete [] lpc->Zs;
        if( lpc->Zss ) delete [] lpc->Zss;
        if( lpc->fit ) delete [] lpc->fit;
        lpc->fit = new double[3 * ( order + 1 )];
        lpc->Zs = new float[order];
        lpc->Zss = new float[order];
        memset( lpc->Zss, 0, order * sizeof(float) );
        lpc->order = order;
        lpc->ticker = 0;
    }

    // autocorrelation r(0..len-1) into lpc->corr; also returns the pitch period
    *pitch = autocorrelate_range( x, len, lpc->corr, lpc->min_lag, lpc->max_lag );

    // The LPC fit uses its own autocorrelation, of the frame under a Hann
    // window (only lags 0..order are needed). The plain autocorrelation
    // above is right for finding the pitch, but fitting to it treats the
    // frame as if it started and stopped abruptly, which smears the
    // spectrum: the fitted formant peaks come out too broad and low, so a
    // filter made from them doesn't fully take the vocal tract out of the
    // residual (and a formant shift then only partly moves the formants).
    double * r = lpc->fit;                   // r[0 .. order]
    double * a = lpc->fit + ( order + 1 );   // a[1 .. order]
    double * t = lpc->fit + 2 * ( order + 1 );
    for( j = 0; j <= order; j++ )
    {
        double sum = 0.0;
        for( i = j; i < len; i++ )
            sum += (double)( x[i] * lpc->win[i] ) * ( x[i-j] * lpc->win[i-j] );
        r[j] = sum;
    }
    // a white-noise floor 50 dB down keeps the fit well-conditioned when
    // part of the spectrum is nearly empty
    r[0] *= 1.0 + 1e-5;

    // solve the "normal equations" sum_j a_j r(|i-j|) = r(i), i = 1..order,
    // by the Levinson-Durbin recursion: O(order^2), and the filter it gives
    // is always stable. (This replaces a general matrix inversion of the
    // Toeplitz matrix R[i][j] = r(|i-j|).)
    for( i = 0; i <= order; i++ ) a[i] = 0.0;
    double err = r[0];
    for( int m = 1; m <= order && err > 0.0; m++ )
    {
        double acc = r[m];
        for( i = 1; i < m; i++ ) acc -= a[i] * r[m-i];
        const double k = acc / err;          // reflection coefficient
        for( i = 1; i < m; i++ ) t[i] = a[i] - k * a[m-i];
        for( i = 1; i < m; i++ ) a[i] = t[i];
        a[m] = k;
        err *= 1.0 - k * k;
    }
    for( i = 0; i < order; i++ )
        coefs[i] = (float)a[i+1];

    // run the prediction filter with the new coefficients to get the
    // residue and the gain
    *power = lpc_predict( lpc, x, len, coefs, order, residue );
}




//-----------------------------------------------------------------------------
// name: lpc_excitation()
// desc: the excitation part of LPC synthesis: writes len samples of the
//       signal that drives the all-pole filter:
//         pitch == 0 (unvoiced)  white noise
//         pitch  > 0 (voiced)    one pulse every `pitch` samples: a single
//                                impulse (impulse train), or, if alt is
//                                non-zero and an alt waveform is loaded, a
//                                copy of the stored glottal pulse
//       `power` scales it. The pulse timing (ticker) persists between calls,
//       so consecutive calls produce one continuous pulse train.
//-----------------------------------------------------------------------------
void lpc_excitation( lpc_data lpc, SAMPLE * e, int len, float power, float pitch, int alt )
{
    SAMPLE output;
    int i, j;

    for( i = 0; i < len; i++ ) {
        output = 0.0f;

        if( pitch == 0 )
        {
            // UNVOICED: uniform white noise in [-1, 1] scaled by the gain
            // (the 20x is an empirical loudness match to the voiced case)
            output = power * 20.0f * ( 2.0f * rand() / RAND_MAX - 1.0f );
            // restart the pulse train cleanly when voicing resumes
            lpc->ticker = 0;
        }
        else {
            // VOICED: count down to the next pitch pulse
            lpc->ticker--;
            if( lpc->ticker <= 0 ) {
                // start a new period (the period rounded to whole samples)
                lpc->ticker = (int)(pitch + .5f);
                // impulse-train excitation: one impulse per period. Scaling
                // by the period makes up for there being fewer pulses per
                // second at lower pitches.
                if( !alt || !lpc->alt )
                    output = power * pitch * 1.0f;
            }

            if( alt && lpc->alt )
            {
                // glottal-pulse excitation. j = samples since the current
                // period began (0 at the pulse onset). The stored pulse is
                // read at every 4th sample, i.e. played 4x faster than it
                // was recorded, and scaled from 16-bit values to +/-1. If the
                // period is shorter than the pulse, the pulse is cut off at
                // the next onset; past its end the output is 0 (output was
                // reset to 0 above, so the "*= .9f" has no effect).
                j = (int)(pitch+.5) - lpc->ticker + 0;
                if( j >= 0 && (j*4) < lpc->alt_len )
                    output = power * pitch * .5f * lpc->alt[j*4] / (float)SHRT_MAX;
                else output *= .9f;
            }
        }

        e[i] = output;
    }
}




//-----------------------------------------------------------------------------
// name: lpc_filter()
// desc: the filter part of LPC synthesis: runs e[0..len-1] through the
//       all-pole filter
//           y[i] = e[i] + sum_k coefs[k] * y[i-1-k]
//       into y (which may be e). state holds the filter's last `order`
//       outputs, newest first; it's read at the start and updated, so the
//       caller decides where the filter's history comes from.
//-----------------------------------------------------------------------------
void lpc_filter( const SAMPLE * e, SAMPLE * y, int len, const float * coefs, int order, float * state )
{
    for( int i = 0; i < len; i++ )
    {
        SAMPLE output = e[i];
        for( int j = 0; j < order; j++ )
            output += state[j] * coefs[j];
        for( int j = order - 1; j > 0; j-- )
            state[j] = state[j-1];
        state[0] = output;
        y[i] = output;
    }
}




//-----------------------------------------------------------------------------
// name: lpc_synthesize()
// desc: LPC synthesis: lpc_excitation() through the all-pole filter defined
//       by coefs, into y[0..len-1]. The pulse timing (ticker) and filter
//       history (Zss) persist between calls so consecutive buffers join up.
//       In unvoiced buffers the filter history is cleared at the first and
//       last sample, so ringing from a voiced buffer (with different
//       coefficients) doesn't carry into or out of the noise.
//-----------------------------------------------------------------------------
void lpc_synthesize( lpc_data lpc, SAMPLE * y, int len, float * coefs,
                     int order, float power, float pitch, int alt )
{
    lpc_excitation( lpc, y, len, power, pitch, alt );

    if( pitch == 0 && len > 0 )
    {
        memset( lpc->Zss, 0, lpc->order * sizeof(float) );
        lpc_filter( y, y, len - 1, coefs, order, lpc->Zss );
        memset( lpc->Zss, 0, lpc->order * sizeof(float) );
        lpc_filter( y + len - 1, y + len - 1, 1, coefs, order, lpc->Zss );
    }
    else
        lpc_filter( y, y, len, coefs, order, lpc->Zss );
}




//-----------------------------------------------------------------------------
// name: autocorrelate_range()
// desc: computes the autocorrelation of x[0..len-1] into y[0..len-1]
//       (y[n] = how well the signal matches itself shifted by n samples),
//       then uses it to estimate the pitch period.
//       Returns the period in samples, or 0 if the frame looks unvoiced.
//       It looks for periods between min_lag and max_lag samples (0 = the
//       defaults below). max_lag is capped at len/2, so that at least one whole
//       period of the signal overlaps itself at the lag being tested.
//-----------------------------------------------------------------------------
// default range, as fractions of the frame length: about 60 .. 1200 Hz for
// 1024-sample frames at 24 kHz
#define PITCH_MIN_LAG_FRACTION  0.02
#define PITCH_MAX_LAG_FRACTION  0.40
// a frame is voiced if, at the detected period, the signal correlates with
// its shifted self at least this well (normalized correlation, 0 .. 1)
#define VOICING_THRESHOLD       0.5

float autocorrelate_range( SAMPLE * x, int len, SAMPLE * y, float min_lag, float max_lag )
{
    float norm, temp;
    int n, i, j, k;

    // raw autocorrelation: y[n] = sum over i of x[i] * x[i+n]
    // (the "- 1" in the inner bound leaves out the last product of each
    // sum; slightly off from the textbook formula, but harmless)
    // original note: "refer to pp. 89 for variable name consistency"
    for ( n = 0; n < len; n++ )
    {
        temp = 0.0;
        for ( i = 0; i < len - n - 1; i++ )
            temp += x[i] * x[i+n];
        y[n] = temp;
    }

    // --- pitch detection ---
    // Every signal correlates strongly with itself at small lags, so the
    // autocorrelation starts with a big peak at lag 0 and falls off. The
    // pitch period shows up as the next big peak after that.
    int lo = min_lag > 0 ? (int)min_lag : (int)( len * PITCH_MIN_LAG_FRACTION );
    int hi = max_lag > 0 ? (int)( max_lag + 0.999f ) : (int)( len * PITCH_MAX_LAG_FRACTION );
    if( hi > len / 2 ) hi = len / 2;
    if( lo < 2 ) lo = 2;
    j = 0;

    if( y[0] > 0.0f && lo < hi )
    {
        // walk forward from the shortest period while the autocorrelation
        // keeps falling, to get past the lag-0 peak into the first valley
        temp = y[0];
        j = lo;
        while( j < hi && y[j] < temp )
        {
            temp = y[j];
            j++;
        }

        // the highest peak after that valley, up to the longest period, is
        // the candidate. (Each y[n] sums fewer products as n grows, which
        // slightly favors shorter lags: a peak at twice the period never
        // beats the period itself, so the pitch isn't halved by mistake.)
        temp = 0.0f;
        for( i = j; i <= hi; i++ )
        {
            if( y[i] > temp )
            {
                j = i;
                temp = y[i];
            }
        }

        // voicing decision, on the normalized correlation at that lag:
        //   sum x[i] x[i+j] / sqrt( sum x[i]^2 * sum x[i+j]^2 )
        // over the samples that overlap. 1 for a perfectly periodic signal at
        // any period. (Comparing y[j] / y[0] instead, as this code used to,
        // penalizes long periods: y[j] has fewer terms, and the taper below
        // shrinks it again, so a perfectly periodic 60 Hz voice in a
        // 1024-sample frame at 24 kHz would only score about 0.37.)
        double xy = 0.0, xx = 0.0, yy = 0.0;
        for( i = 0; i + j < len; i++ )
        {
            xy += (double)x[i] * x[i + j];
            xx += (double)x[i] * x[i];
            yy += (double)x[i + j] * x[i + j];
        }
        if( temp <= 0.0f || !( xx * yy > 0.0 ) || xy / sqrt( xx * yy ) < VOICING_THRESHOLD )
            j = 0;
        // a peak right at the longest period allowed most likely belongs to
        // a period beyond it: below the range, so not reported as a pitch
        if( j >= hi && hi < len / 2 ) j = 0;
    }

    // taper the autocorrelation by (len - lag) / len, a triangular "lag
    // window". It isn't a normalization in the usual sense (each sum
    // already shrinks with lag because it has fewer terms); its effect on
    // the LPC fit is a mild smoothing. y is what lpc_analyze builds R from.
    norm = 1.0f / len;
    k = len;

    for( i = 0; i < len; i++ )
        y[i] *= (k-i) * norm;

    // pitch period in samples (0 = unvoiced, including silence)
    return (float) j;
}




//------------------------------------------------------------------------------
// name: lpc_preemphasis()
// desc: pre-emphasis, in place: x[i] = x[i] - alpha * x[i-1], a first-order
//       high-pass that boosts high frequencies. Speech energy falls off
//       with frequency, so tilting the spectrum up first lets the LPC fit
//       the upper formants better. Runs backward so each step still sees
//       the original x[i-1]. x[0] is unchanged, and no state is kept
//       between buffers.
//------------------------------------------------------------------------------
void lpc_preemphasis( SAMPLE * x, int len, float alpha )
{
    for( int i = len - 1; i > 0; i-- )
        x[i] = x[i] - alpha * x[i-1];
}




//------------------------------------------------------------------------------
// name: lpc_deemphasis()
// desc: de-emphasis, in place: x[i] = alpha * x[i] + (1 - alpha) * x[i-1],
//       a 2-tap low-pass (with alpha = 0.5, the average of neighboring
//       samples). It tilts the spectrum back down after lpc_preemphasis
//       but is NOT its exact inverse, which would be the recursive filter
//       y[i] = x[i] + alpha * y[i-1].
//------------------------------------------------------------------------------
void lpc_deemphasis( SAMPLE * x, int len, float alpha )
{
    for( int i = len - 1; i > 0; i-- )
        x[i] = alpha * x[i] + (1.0f-alpha) * x[i-1];
}




//-----------------------------------------------------------------------------
// name: lpc_alt()
// desc: installs a copy of alt[0..len-1] as the alternate (glottal pulse)
//       excitation for voiced synthesis. Passing NULL or 0 removes it, and
//       lpc_synthesize() then falls back to the impulse train.
//-----------------------------------------------------------------------------
void lpc_alt( lpc_data lpc, SAMPLE * alt, int len )
{
    if( lpc->alt )
    {
        delete [] lpc->alt;
        lpc->alt = NULL;
        lpc->alt_len = 0;
    }
    
    if( !alt || !len ) return;

    lpc->alt = new SAMPLE[len];
    memcpy( lpc->alt, alt, len * sizeof(SAMPLE) );
    lpc->alt_len = len;
}




//=============================================================================
// Reflection coefficients
//=============================================================================

//-----------------------------------------------------------------------------
// name: lpc_to_reflection()
// desc: predictor coefficients -> reflection coefficients, by running the
//       Levinson-Durbin recursion backwards ("step-down"):
//           k_m = a_m^(m)
//           a_i^(m-1) = ( a_i^(m) + k_m a_(m-i)^(m) ) / ( 1 - k_m^2 )
//-----------------------------------------------------------------------------
bool lpc_to_reflection( const float * coefs, int order, float * k )
{
    double a[LPC_REFLECTION_MAX_ORDER], t[LPC_REFLECTION_MAX_ORDER];
    bool stable = true;
    for( int i = 0; i < order; i++ ) a[i] = coefs[i];
    for( int m = order; m >= 1; m-- )
    {
        double km = a[m-1];
        if( !std::isfinite( km ) ) { km = 0; stable = false; }
        else if( !( fabs( km ) < 1.0 ) || fabs( km ) > .999 )
        {
            if( !( fabs( km ) < 1.0 ) ) stable = false;
            km = km > 0 ? .999 : -.999;
        }
        k[m-1] = (float)km;
        const double d = 1.0 - km * km;
        for( int i = 1; i < m; i++ )
            t[i-1] = ( a[i-1] + km * a[m-i-1] ) / d;
        for( int i = 0; i < m - 1; i++ ) a[i] = t[i];
    }
    return stable;
}




//-----------------------------------------------------------------------------
// name: reflection_to_lpc()
// desc: reflection coefficients -> predictor coefficients (the Levinson-
//       Durbin "step-up"):
//           a_m^(m) = k_m
//           a_i^(m) = a_i^(m-1) - k_m a_(m-i)^(m-1)
//-----------------------------------------------------------------------------
void reflection_to_lpc( const float * k, int order, float * coefs )
{
    double a[LPC_REFLECTION_MAX_ORDER], t[LPC_REFLECTION_MAX_ORDER];
    for( int m = 1; m <= order; m++ )
    {
        const double km = k[m-1];
        for( int i = 1; i < m; i++ )
            t[i-1] = a[i-1] - km * a[m-i-1];
        for( int i = 0; i < m - 1; i++ ) a[i] = t[i];
        a[m-1] = km;
    }
    for( int i = 0; i < order; i++ ) coefs[i] = (float)a[i];
}





//=============================================================================
// Formants: measuring them and shifting them
//
// Three ways, each with a measuring function and a shifting ("warp")
// function:
//   envelope  lpc_formants() / warp_formants(): on the sampled spectral
//             envelope
//   poles     lpc_formants_poles() / warp_formants_poles(): on the roots of
//             A(z)
//   LSFs      lpc_formants_lsf() / warp_formants_lsf(): on the line spectral
//             frequencies
// plus lpc_measure_formants(), a separate measurement of a frame of speech
// (not of a given filter) for reporting.
//=============================================================================
#define FORMANT_MIN_HZ  90.0        // ignore envelope peaks below this...
#define FORMANT_MAX_HZ  5500.0      // ...and above this
#define FORMANT_MAX_BW  1000.0      // broader peaks than this (Hz) are spectral
                                    //   tilt or noise, not formants

// Number of points around the unit circle at which warp_formants() samples
// the spectral envelope (must be even). The spacing has to be fine enough to
// resolve the sharpest formant peaks: 2048 points is ~12 Hz at 24 kHz.
#define WARP_GRID 2048

//-----------------------------------------------------------------------------
// name: envelope_power()
// desc: power of the LPC synthesis filter's frequency response at angle w
//       (radians per sample, 0 .. pi = 0 Hz .. Nyquist):
//           1 / |A(e^jw)|^2,  A(z) = 1 - a[0] z^-1 - ... - a[order-1] z^-order
//       This is the smooth spectral envelope that LPC fits to the voice;
//       its peaks are the formants.
//-----------------------------------------------------------------------------
static double envelope_power( const double * a, int order, double w )
{
    // A(e^jw) = 1 - sum over m of a[m-1] e^(-jmw). cos(mw) and sin(mw) are
    // generated by rotating one step of w at a time.
    const double cw = cos( w ), sw = sin( w );
    double c = 1.0, s = 0.0;      // cos(mw), sin(mw), starting at m = 0
    double re = 1.0, im = 0.0;    // A(e^jw)
    for( int m = 0; m < order; m++ )
    {
        // advance to angle (m+1) w
        double c2 = c * cw - s * sw;
        s = s * cw + c * sw;
        c = c2;
        // e^(-j(m+1)w) = c - j s
        re -= a[m] * c;
        im += a[m] * s;
    }

    // floor |A|^2 so a pole sitting right on the grid can't divide by zero
    double mag2 = re * re + im * im;
    return 1.0 / ( mag2 > 1e-12 ? mag2 : 1e-12 );
}




//-----------------------------------------------------------------------------
// name: find_formants()
// desc: picks the formants out of a sampled LPC envelope. env[0..H] is the
//       envelope's power at w = pi*k/H (0 Hz .. Nyquist) and fs the sample
//       rate. Each formant is a local peak of the envelope:
//         - frequency: the peak's grid position, refined by fitting a
//           parabola through the peak and its two neighbors (in dB). This
//           resolves much finer than the grid spacing.
//         - bandwidth: the distance between the points 3 dB below the peak,
//           found by walking outward from the peak on each side
//       A peak is kept only if it lies between FORMANT_MIN_HZ and
//       FORMANT_MAX_HZ, falls at least 3 dB on some side before the envelope
//       starts rising toward the next peak (anything shallower is a
//       shoulder, not a resonance), and is narrower than FORMANT_MAX_BW.
//       Limitation: two formants closer together than their bandwidths
//       merge into one peak, so the second is missed and the later ones
//       get the wrong F-numbers. Finding the roots of A(z) avoids that.
//-----------------------------------------------------------------------------
static void find_formants( const double * env, int H, double fs, FormantSet * out )
{
    static double db[WARP_GRID / 2 + 1];   // envelope in dB (static: not thread-safe)
    const double hz_per_bin = fs / ( 2.0 * H );
    int k, j;

    out->count = 0;
    if( H > WARP_GRID / 2 ) return;

    for( k = 0; k <= H; k++ )
        db[k] = 10.0 * log10( env[k] );

    for( k = 1; k < H && out->count < LPC_MAX_FORMANTS; k++ )
    {
        // local maximum? (">=" on the right so a flat top counts only once)
        if( !( db[k] > db[k-1] && db[k] >= db[k+1] ) ) continue;

        if( k * hz_per_bin < FORMANT_MIN_HZ ) continue;
        if( k * hz_per_bin > FORMANT_MAX_HZ ) break;

        // vertex of the parabola through the peak and its neighbors:
        // offset delta (in bins, within +/- 0.5) and height peak_db
        double l = db[k-1], c = db[k], r = db[k+1];
        double denom = l - 2.0 * c + r;
        double delta = denom < 0.0 ? 0.5 * ( l - r ) / denom : 0.0;
        double peak_db = c - 0.25 * ( l - r ) * delta;
        double freq = ( k + delta ) * hz_per_bin;

        // walk outward to the -3 dB points, interpolating between grid
        // points. Give up on a side if the envelope starts rising again
        // first (we've hit the valley before a neighboring peak).
        double edge = peak_db - 3.0;
        double lo = -1.0, hi = -1.0;           // -1 = not reached
        for( j = k; j > 0; j-- )
        {
            if( db[j-1] > db[j] ) break;
            if( db[j-1] <= edge )
            {
                lo = ( j - 1 + ( edge - db[j-1] ) / ( db[j] - db[j-1] ) ) * hz_per_bin;
                break;
            }
        }
        for( j = k; j < H; j++ )
        {
            if( db[j+1] > db[j] ) break;
            if( db[j+1] <= edge )
            {
                hi = ( j + ( db[j] - edge ) / ( db[j] - db[j+1] ) ) * hz_per_bin;
                break;
            }
        }

        // bandwidth from both edges, or twice the half-width if only one
        // side gets down 3 dB before the next peak
        double bw;
        if( lo >= 0.0 && hi >= 0.0 ) bw = hi - lo;
        else if( lo >= 0.0 ) bw = 2.0 * ( freq - lo );
        else if( hi >= 0.0 ) bw = 2.0 * ( hi - freq );
        else continue;                          // a shoulder, not a formant

        if( bw > FORMANT_MAX_BW ) continue;

        Formant & f = out->f[out->count++];
        f.freq = freq;
        f.bw = bw;
        f.level = peak_db;
    }
}




//-----------------------------------------------------------------------------
// name: lpc_formants()
// desc: finds the formants of an LPC filter straight from its coefficients:
//       samples the envelope and runs find_formants() on it. Use this when
//       you want the formants without warping anything.
//-----------------------------------------------------------------------------
void lpc_formants( const float * coefs, int order, float srate, FormantSet * out )
{
    const int H = WARP_GRID / 2;
    static double env[WARP_GRID / 2 + 1];   // static: not thread-safe
    double a[1024];
    int i, k;

    out->count = 0;
    if( order < 1 || order > 1023 ) return;

    for( i = 0; i < order; i++ )
        a[i] = coefs[i];
    for( k = 0; k <= H; k++ )
        env[k] = envelope_power( a, order, LPC__PI * k / H );

    find_formants( env, H, srate, out );
}




//-----------------------------------------------------------------------------
// name: warp_formants()
// desc: moves every formant by the factor alpha (> 1 raises them, like a
//       shorter vocal tract; < 1 lowers them) by warping the LPC spectral
//       envelope along the frequency axis and fitting a new all-pole model
//       to the result:
//         1. sample the warped envelope S(w) = 1 / |A(e^(j w/alpha))|^2 on a
//            grid around the unit circle. Whatever was at frequency w0 in
//            the original envelope ends up at alpha * w0.
//         2. inverse Fourier transform S to get the autocorrelation r(0..order)
//            of a signal with that spectrum
//         3. solve for new coefficients from r with the Levinson-Durbin
//            recursion (the same normal equations lpc_analyze() solves).
//            A valid autocorrelation always gives a stable filter.
//         4. rescale *power so the new filter passes the same average power
//            as the old one, so the shift doesn't change the loudness
//       For alpha < 1 the top of the band would come from above Nyquist,
//       which doesn't exist; it reuses the envelope value at Nyquist.
//       coefs and *power are left unchanged if the refit fails numerically.
//       Optionally measures the formants (see find_formants()) along the way,
//       from envelopes the warp computes anyway:
//         before  the original envelope: what the analysis found
//         after   the refit filter's envelope: what will be synthesized.
//                 If the refit fails this is the same as before, since the
//                 coefficients are unchanged.
//       Either may be NULL. srate is the sample rate the coefficients are
//       for (used to give the formants in Hz). Not thread-safe (static
//       work buffers).
//-----------------------------------------------------------------------------
void warp_formants( float * coefs, int order, float srate, float alpha, float * power,
                    FormantSet * before, FormantSet * after )
{
    const int M = WARP_GRID;          // grid points around the whole circle
    const int H = WARP_GRID / 2;      // grid index of Nyquist (w = pi)

    static double cos_table[WARP_GRID];   // cos( 2 pi j / M )
    static bool table_ready = false;
    static double S[WARP_GRID / 2 + 1];   // warped envelope, 0 .. Nyquist
    static double S_old[WARP_GRID / 2 + 1];  // original envelope, 0 .. Nyquist
    static double S_new[WARP_GRID / 2 + 1];  // refit filter's envelope, 0 .. Nyquist
    double a[1024];                       // original coefficients, in double
    double r[1024];                       // autocorrelation of the warped envelope
    double A[1025], tmp[1025];            // Levinson-Durbin solution, A[1..order]
    int i, j, k, n;

    if( before ) before->count = 0;
    if( after ) after->count = 0;
    if( order < 1 || order > 1023 ) return;

    if( !table_ready )
    {
        for( j = 0; j < M; j++ )
            cos_table[j] = cos( 2.0 * LPC__PI * j / M );
        table_ready = true;
    }

    for( i = 0; i < order; i++ )
        a[i] = coefs[i];

    // 1. warped envelope on the upper half circle (0 .. Nyquist; the lower
    //    half is its mirror image). Also total up the ORIGINAL envelope over
    //    the whole circle: its average is the old filter's power gain,
    //    needed in step 4.
    double old_gain = 0.0;
    for( k = 0; k <= H; k++ )
    {
        double w = LPC__PI * k / H;
        double src = w / alpha;
        if( src > LPC__PI ) src = LPC__PI;
        S[k] = envelope_power( a, order, src );

        // points strictly between 0 and pi stand for themselves and their
        // mirror image, so they count twice
        double p = envelope_power( a, order, w );
        S_old[k] = p;
        old_gain += ( k == 0 || k == H ) ? p : 2.0 * p;
    }
    old_gain /= M;

    // formants of the original envelope. "after" starts out the same, in
    // case the refit below fails and the coefficients stay as they are.
    if( before || after )
    {
        FormantSet orig;
        find_formants( S_old, H, srate, &orig );
        if( before ) *before = orig;
        if( after ) *after = orig;
    }

    // 2. autocorrelation = inverse DFT of the power spectrum. S is real and
    //    symmetric, so only cosines remain:
    //      r(n) = ( S(0) + S(pi) (-1)^n + 2 sum_{k=1}^{H-1} S(k) cos(2 pi k n / M) ) / M
    for( n = 0; n <= order; n++ )
    {
        double sum = S[0] + ( ( n & 1 ) ? -S[H] : S[H] );
        for( k = 1; k < H; k++ )
            sum += 2.0 * S[k] * cos_table[ ( k * n ) % M ];
        r[n] = sum / M;
    }

    // 3. Levinson-Durbin: solve sum_j A[j] r(|i-j|) = r(i), i = 1..order,
    //    one order at a time. E is the prediction error at the current order
    //    and refl the reflection coefficient; |refl| < 1 at every step means
    //    the filter is stable, so bail out if rounding ever breaks that.
    double E = r[0];
    if( !( E > 0.0 ) ) return;
    for( i = 1; i <= order; i++ )
    {
        double acc = r[i];
        for( j = 1; j < i; j++ )
            acc -= A[j] * r[i - j];
        double refl = acc / E;
        if( !( fabs( refl ) < 1.0 ) ) return;

        tmp[i] = refl;
        for( j = 1; j < i; j++ )
            tmp[j] = A[j] - refl * A[i - j];
        for( j = 1; j <= i; j++ )
            A[j] = tmp[j];

        E *= ( 1.0 - refl * refl );
    }

    // 4. loudness: the new filter's average power over the whole circle,
    //    computed the same way as old_gain
    double new_gain = 0.0;
    for( k = 0; k <= H; k++ )
    {
        double p = envelope_power( A + 1, order, LPC__PI * k / H );
        S_new[k] = p;
        new_gain += ( k == 0 || k == H ) ? p : 2.0 * p;
    }
    new_gain /= M;
    if( !( old_gain > 0.0 && old_gain < 1e30 && new_gain > 0.0 && new_gain < 1e30 ) )
        return;

    // success: hand back the new coefficients and the loudness-matched gain
    for( i = 0; i < order; i++ )
        coefs[i] = (float)A[i + 1];
    *power *= (float)sqrt( old_gain / new_gain );

    // formants of the filter that will actually be synthesized
    if( after )
        find_formants( S_new, H, srate, after );
}




//=============================================================================
// Two more ways to shift and measure formants, working on representations
// of the LPC filter in which the formants are explicit, instead of on the
// sampled envelope like warp_formants() / lpc_formants():
//
//   poles   the roots of A(z). Each complex-conjugate pair of poles
//           r * e^(+/- j theta) is one resonance, with
//               frequency = theta * fs / (2 pi)
//               bandwidth = -ln(r) * fs / pi
//           Frequency and bandwidth can be changed independently, and the
//           measured formants are exact pole positions. Needs a polynomial
//           root finder.
//
//   LSFs    line spectral frequencies: A(z) split into a symmetric P(z) and
//           an antisymmetric Q(z) whose roots all lie on the unit circle, at
//           `order` interleaved frequencies. Formants show up as pairs of
//           closely spaced LSFs. The filter is stable exactly when the LSFs
//           are strictly increasing in (0, pi), so they're the safest
//           representation to edit. This is what speech codecs quantize.
//
// Both warps end with the same final stage, finish_warp(), for two reasons
// found in testing:
//   - Moving poles or LSFs moves the formants accurately, but it also
//     changes the overall spectral tilt, often by tens of dB: at LPC orders
//     like 30, most poles and LSFs aren't formants but jointly shape the
//     spectral slope and the high-frequency floor, and moving them breaks
//     that balance. The output would sound much brighter (alpha > 1) or
//     duller (alpha < 1) than intended. finish_warp() restores the
//     original's slope (and for LSFs its broad curvature too) through its
//     lowest cepstral coefficients, which leaves the formant peaks where
//     the edit put them.
//   - The edited polynomial's coefficients can be large and badly
//     conditioned, and lpc_synthesize() runs in float: rounding them can
//     make the filter unstable. finish_warp() refits the coefficients with
//     the Levinson-Durbin recursion, which always gives well-conditioned,
//     stable ones.
//
// All functions follow the conventions of warp_formants(): coefs are the
// predictor coefficients from lpc_analyze(), A(z) = 1 - sum a_k z^-k;
// coefs and *power are left alone if anything fails numerically; `before`
// and `after` (optional) receive the formants of the original filter and
// of the filter that will be synthesized, measured with the method's own
// lpc_formants_*() function. Not thread-safe (static work buffers).
//=============================================================================

// Minimal complex number type (all the root finder needs)
struct Cx { double re, im; };
static inline Cx cx( double re, double im ) { Cx c; c.re = re; c.im = im; return c; }
static inline Cx operator+( Cx a, Cx b ) { return cx( a.re + b.re, a.im + b.im ); }
static inline Cx operator-( Cx a, Cx b ) { return cx( a.re - b.re, a.im - b.im ); }
static inline Cx operator*( Cx a, Cx b )
{ return cx( a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re ); }
static inline Cx operator/( Cx a, Cx b )
{
    double d = b.re * b.re + b.im * b.im;
    return cx( ( a.re * b.re + a.im * b.im ) / d, ( a.im * b.re - a.re * b.im ) / d );
}
static inline double cabs2( Cx a ) { return a.re * a.re + a.im * a.im; }

// pole radius cap after editing, to keep the filter safely stable
#define POLE_MAX_RADIUS     0.9995
// highest angle a formant pole may be moved to (95% of Nyquist)
#define POLE_MAX_ANGLE      ( LPC__PI * 0.95 )
// fraction of Nyquist up to which warp_angle() scales frequencies exactly
// by alpha (see there)
#define WARP_KNEE           0.8
// number of cepstral coefficients (1 = spectral slope, 2 and 3 = broad
// curvature) that finish_warp() restores to their original values. The
// pole warp only needs the slope; restoring more pulls on the formants it
// just placed. The LSF warp disturbs the broad shape more and needs three.
#define MAX_BALANCE_TERMS   3
#define POLE_BALANCE_TERMS  1
#define LSF_BALANCE_TERMS   3
// imaginary parts smaller than this count as real roots
#define POLE_REAL_EPS       1e-8
// points on (0, pi) searched for sign changes when finding LSFs. Has to be
// fine enough that no two LSFs share an interval: 4096 is ~3 Hz at 24 kHz.
#define LSF_GRID            4096
// smallest spacing allowed between edited LSFs, in Hz
#define LSF_MIN_GAP_HZ      10.0




//-----------------------------------------------------------------------------
// name: warp_angle()
// desc: the frequency map used by the LSF warp, on angles in
//       radians (0 .. pi = 0 Hz .. Nyquist). Below a knee it is exactly
//       w * alpha; above it, the rest of the band is mapped linearly onto
//       whatever is left up to Nyquist:
//         alpha > 1: the knee is at WARP_KNEE * pi / alpha, so it lands at
//                    WARP_KNEE * pi, and the band above is squeezed into the
//                    remaining 20%. Nothing is pushed past Nyquist.
//         alpha < 1: the knee is at WARP_KNEE * pi, and the band above is
//                    stretched up to Nyquist, so the top isn't left empty.
//       Strictly increasing, and maps (0, pi) into (0, pi), so the order of
//       the LSFs is preserved. The knee is well above the formants
//       (WARP_KNEE * Nyquist is 9.6 kHz at 24 kHz) unless alpha is large:
//       at alpha = 2 it's 4.8 kHz, and F4 and up get squeezed too.
//       (The envelope warp handles this differently; see warp_formants().)
//-----------------------------------------------------------------------------
static double warp_angle( double w, double alpha )
{
    const double knee = WARP_KNEE * LPC__PI / ( alpha > 1.0 ? alpha : 1.0 );
    if( w <= knee ) return alpha * w;
    // straight line from (knee, alpha * knee) to (pi, pi)
    return alpha * knee + ( w - knee ) * ( LPC__PI - alpha * knee ) / ( LPC__PI - knee );
}




//-----------------------------------------------------------------------------
// name: lpc_power_gain()
// desc: average power gain of the synthesis filter 1/A(z) over the whole
//       unit circle, i.e. its output power for white-noise input. Used to
//       keep loudness constant across a formant shift.
//-----------------------------------------------------------------------------
static double lpc_power_gain( const double * a, int order )
{
    const int H = WARP_GRID / 2;
    double g = 0.0;
    for( int k = 0; k <= H; k++ )
    {
        double p = envelope_power( a, order, LPC__PI * k / H );
        g += ( k == 0 || k == H ) ? p : 2.0 * p;   // interior points count twice
    }
    return g / WARP_GRID;
}




//-----------------------------------------------------------------------------
// name: envelope_to_lpc()
// desc: fits predictor coefficients a[0..order-1] to a power spectrum
//       S[0..WARP_GRID/2] (sampled like the envelopes in warp_formants():
//       w = pi*k/H): inverse DFT to an autocorrelation, then Levinson-Durbin.
//       Same as steps 2 and 3 of warp_formants(). Returns false if the
//       recursion breaks down numerically; otherwise the filter is stable.
//-----------------------------------------------------------------------------
static bool envelope_to_lpc( const double * S, int order, double * a )
{
    const int M = WARP_GRID, H = WARP_GRID / 2;
    static double cos_table[WARP_GRID];
    static bool table_ready = false;
    double r[1025], A[1025], tmp[1025];
    int i, j, k, n;

    if( order < 1 || order > 1023 ) return false;
    if( !table_ready )
    {
        for( j = 0; j < M; j++ )
            cos_table[j] = cos( 2.0 * LPC__PI * j / M );
        table_ready = true;
    }

    // autocorrelation: inverse DFT of the real, symmetric power spectrum
    for( n = 0; n <= order; n++ )
    {
        double sum = S[0] + ( ( n & 1 ) ? -S[H] : S[H] );
        for( k = 1; k < H; k++ )
            sum += 2.0 * S[k] * cos_table[ ( k * n ) % M ];
        r[n] = sum / M;
    }

    // Levinson-Durbin
    double E = r[0];
    if( !( E > 0.0 && E < 1e300 ) ) return false;
    for( i = 1; i <= order; i++ )
    {
        double acc = r[i];
        for( j = 1; j < i; j++ )
            acc -= A[j] * r[i - j];
        double refl = acc / E;
        if( !( fabs( refl ) < 1.0 ) ) return false;
        tmp[i] = refl;
        for( j = 1; j < i; j++ )
            tmp[j] = A[j] - refl * A[i - j];
        for( j = 1; j <= i; j++ )
            A[j] = tmp[j];
        E *= ( 1.0 - refl * refl );
    }

    for( i = 0; i < order; i++ )
        a[i] = A[i + 1];
    return true;
}




//-----------------------------------------------------------------------------
// name: finish_warp()
// desc: the final stage of the pole and LSF warps (see the comment at the
//       top of this section). Given the original coefficients a_orig and
//       the edited ones a_edit:
//         1. sample both envelopes
//         2. compute the lowest cepstral coefficients of each
//            (the cosine-series coefficients of the log envelope; c1 is the
//            overall slope, c2 and c3 broad curvature; `terms` of them,
//            at most MAX_BALANCE_TERMS) and multiply the edited envelope by
//            exp(2 * sum_k (c_orig[k] - c_edit[k]) cos(k w)),
//            which gives it the original's slope and curvature but changes
//            nothing finer, so the formants stay where the edit put them
//         3. refit coefficients to the result (envelope_to_lpc())
//         4. rescale *power so the loudness matches the original
//       Writes the result to coefs (as float) and returns true; leaves coefs
//       and *power alone and returns false if anything fails numerically.
//-----------------------------------------------------------------------------
static bool finish_warp( const double * a_orig, const double * a_edit, int order,
                         int terms, float * coefs, float * power )
{
    const int M = WARP_GRID, H = WARP_GRID / 2;
    static double S0[WARP_GRID / 2 + 1], S1[WARP_GRID / 2 + 1];   // static: not thread-safe
    double c0[MAX_BALANCE_TERMS + 1], c1[MAX_BALANCE_TERMS + 1], a_new[1024];
    int j, k;

    if( terms > MAX_BALANCE_TERMS ) terms = MAX_BALANCE_TERMS;

    // 1. envelopes (evaluated from the double-precision coefficients, so
    //    even badly conditioned edited ones are fine here)
    for( j = 0; j <= H; j++ )
    {
        S0[j] = envelope_power( a_orig, order, LPC__PI * j / H );
        S1[j] = envelope_power( a_edit, order, LPC__PI * j / H );
        if( !( S1[j] > 0.0 && S1[j] < 1e300 ) ) return false;
    }

    // 2. low cepstral coefficients of both:
    //    c[k] = ( L(0) + L(pi) (-1)^k + 2 sum_{j=1}^{H-1} L(j) cos(k w_j) ) / M,
    //    with L = log S
    for( k = 1; k <= terms; k++ )
    {
        double sign = ( k & 1 ) ? -1.0 : 1.0;
        c0[k] = log( S0[0] ) + sign * log( S0[H] );
        c1[k] = log( S1[0] ) + sign * log( S1[H] );
        for( j = 1; j < H; j++ )
        {
            double ck = 2.0 * cos( k * LPC__PI * j / H );
            c0[k] += ck * log( S0[j] );
            c1[k] += ck * log( S1[j] );
        }
        c0[k] /= M;
        c1[k] /= M;
    }
    //    ...and put the original's back into the edited envelope
    for( j = 0; j <= H; j++ )
    {
        double w = LPC__PI * j / H, corr = 0.0;
        for( k = 1; k <= terms; k++ )
            corr += 2.0 * ( c0[k] - c1[k] ) * cos( k * w );
        S1[j] *= exp( corr );
        if( !( S1[j] > 0.0 && S1[j] < 1e300 ) ) return false;
    }

    // 3. refit
    if( !envelope_to_lpc( S1, order, a_new ) ) return false;

    // 4. loudness
    double old_gain = lpc_power_gain( a_orig, order );
    double new_gain = lpc_power_gain( a_new, order );
    if( !( old_gain > 0.0 && old_gain < 1e30 && new_gain > 0.0 && new_gain < 1e30 ) )
        return false;

    for( j = 0; j < order; j++ )
        coefs[j] = (float)a_new[j];
    *power *= (float)sqrt( old_gain / new_gain );
    return true;
}




//-----------------------------------------------------------------------------
// name: is_formant()
// desc: whether a resonance at freq Hz with bandwidth bw Hz counts as a
//       formant: inside FORMANT_MIN_HZ .. FORMANT_MAX_HZ and narrower than
//       FORMANT_MAX_BW
//-----------------------------------------------------------------------------
static bool is_formant( double freq, double bw )
{
    return freq >= FORMANT_MIN_HZ && freq <= FORMANT_MAX_HZ &&
           bw > 0.0 && bw <= FORMANT_MAX_BW;
}




//-----------------------------------------------------------------------------
// name: sort_formants()
// desc: puts candidate formants in increasing frequency order and keeps the
//       ones inside FORMANT_MIN_HZ .. FORMANT_MAX_HZ and narrower than
//       FORMANT_MAX_BW, up to LPC_MAX_FORMANTS, so out->f[0] is F1.
//-----------------------------------------------------------------------------
static void sort_formants( Formant * cand, int n, FormantSet * out )
{
    int i, j;
    // insertion sort by frequency (n is at most order / 2)
    for( i = 1; i < n; i++ )
    {
        Formant t = cand[i];
        for( j = i - 1; j >= 0 && cand[j].freq > t.freq; j-- )
            cand[j + 1] = cand[j];
        cand[j + 1] = t;
    }

    out->count = 0;
    for( i = 0; i < n && out->count < LPC_MAX_FORMANTS; i++ )
    {
        if( !is_formant( cand[i].freq, cand[i].bw ) ) continue;
        out->f[out->count++] = cand[i];
    }
}




//-----------------------------------------------------------------------------
// name: lpc_poles()
// desc: finds the poles of 1/A(z), i.e. the roots of the polynomial
//           z^p - a[0] z^(p-1) - ... - a[p-1]
//       with the Aberth-Ehrlich method: Newton's method on every root at
//       once, with each step corrected so the estimates repel each other
//       and converge to different roots. Converges cubically; a few dozen
//       sweeps are typical. LPC poles from lpc_analyze() all lie inside the
//       unit circle, so the starting guesses are spread around a circle of
//       radius 0.9, rotated off the real axis so they aren't conjugate-
//       symmetric (which can stall the iteration). Roots stop being updated
//       once their steps are down at rounding level.
//       Returns false if it didn't converge.
//-----------------------------------------------------------------------------
static bool lpc_poles( const double * a, int order, Cx * z )
{
    const int MAX_SWEEPS = 200;
    double c[1025];                     // monic polynomial, c[0] = 1
    bool done[1024];                    // root k has converged
    int i, j, k, sweep, remaining;

    if( order < 1 || order > 1024 ) return false;
    c[0] = 1.0;
    for( i = 1; i <= order; i++ )
        c[i] = -a[i - 1];

    for( k = 0; k < order; k++ )
    {
        double th = 2.0 * LPC__PI * k / order + 0.4;
        z[k] = cx( 0.9 * cos( th ), 0.9 * sin( th ) );
        done[k] = false;
    }

    remaining = order;
    for( sweep = 0; sweep < MAX_SWEEPS && remaining > 0; sweep++ )
    {
        for( k = 0; k < order; k++ )
        {
            // roots that have converged stay put (they still repel the
            // others through the sum below)
            if( done[k] ) continue;

            // f = poly(z), df = poly'(z), by Horner's rule
            Cx f = cx( 1.0, 0.0 ), df = cx( 0.0, 0.0 );
            for( i = 1; i <= order; i++ )
            {
                df = df * z[k] + f;
                f = f * z[k] + cx( c[i], 0.0 );
            }
            if( cabs2( f ) == 0.0 ) { done[k] = true; remaining--; continue; }
            if( cabs2( df ) == 0.0 ) df = cx( 1e-12, 0.0 );

            // Newton step f/df, corrected by the other roots' repulsion
            Cx ratio = f / df;
            Cx sum = cx( 0.0, 0.0 );
            for( j = 0; j < order; j++ )
                if( j != k ) sum = sum + cx( 1.0, 0.0 ) / ( z[k] - z[j] );
            Cx denom = cx( 1.0, 0.0 ) - ratio * sum;
            Cx step = cabs2( denom ) > 0.0 ? ratio / denom : ratio;

            z[k] = z[k] - step;
            // converged once the step is at the level of rounding error
            if( cabs2( step ) < 1e-28 * ( 1.0 + cabs2( z[k] ) ) )
            {
                done[k] = true;
                remaining--;
            }
        }
    }

    return remaining == 0;
}




//-----------------------------------------------------------------------------
// name: split_poles()
// desc: sorts roots from lpc_poles() into complex-conjugate pairs (stored
//       once, as the member with positive imaginary part, by angle `theta`
//       and radius `rad`) and real roots (`real`, imaginary part dropped).
//       Returns false if the conjugates don't match up in number.
//-----------------------------------------------------------------------------
static bool split_poles( const Cx * z, int order, double * theta, double * rad,
                         int * npairs, double * real, int * nreal )
{
    int upper = 0, lower = 0, i;
    *npairs = *nreal = 0;
    for( i = 0; i < order; i++ )
    {
        if( z[i].im > POLE_REAL_EPS )
        {
            theta[*npairs] = atan2( z[i].im, z[i].re );
            rad[*npairs] = sqrt( cabs2( z[i] ) );
            ( *npairs )++;
            upper++;
        }
        else if( z[i].im < -POLE_REAL_EPS )
            lower++;
        else
            real[( *nreal )++] = z[i].re;
    }
    return upper == lower;
}




//-----------------------------------------------------------------------------
// name: spread_order()
// desc: given n factors sorted by frequency, fills idx[0..n-1] with the
//       order to multiply them in: van der Corput (bit-reversed index)
//       order, which keeps the roots of every partial product spread evenly
//       around the circle. Multiplying in sorted order instead builds up
//       huge intermediate coefficients that later cancel; above order ~50
//       that loses essentially all precision.
//-----------------------------------------------------------------------------
static void spread_order( int n, int * idx )
{
    unsigned int key[1024];
    int i, j, b;
    for( i = 0; i < n; i++ )
    {
        unsigned int r = 0, v = (unsigned int)i;
        for( b = 0; b < 16; b++ ) { r = ( r << 1 ) | ( v & 1 ); v >>= 1; }
        key[i] = r;
        idx[i] = i;
    }
    // insertion sort of idx by key
    for( i = 1; i < n; i++ )
    {
        int t = idx[i];
        for( j = i - 1; j >= 0 && key[idx[j]] > key[t]; j-- )
            idx[j + 1] = idx[j];
        idx[j + 1] = t;
    }
}




//-----------------------------------------------------------------------------
// name: poles_to_lpc()
// desc: rebuilds predictor coefficients a[0..order-1] from conjugate pairs
//       (theta, rad) and real roots by multiplying out
//           (1 - 2 r cos(theta) z^-1 + r^2 z^-2)  for each pair
//           (1 - x z^-1)                          for each real root
//       All in real arithmetic, so the result is exactly real. The pairs
//       are multiplied in spread_order() by angle to keep precision.
//-----------------------------------------------------------------------------
static void poles_to_lpc( const double * theta, const double * rad, int npairs,
                          const double * real, int nreal, double * a )
{
    double poly[1026];
    int byangle[512], spread[512];
    int deg = 0, i, j, k;
    poly[0] = 1.0;

    // pair indices sorted by angle, then put in spread order
    for( i = 0; i < npairs; i++ )
    {
        int t = i;
        for( j = i - 1; j >= 0 && theta[byangle[j]] > theta[t]; j-- )
            byangle[j + 1] = byangle[j];
        byangle[j + 1] = t;
    }
    spread_order( npairs, spread );

    for( int n = 0; n < npairs; n++ )
    {
        i = byangle[spread[n]];
        double b1 = -2.0 * rad[i] * cos( theta[i] ), b2 = rad[i] * rad[i];
        poly[deg + 1] = 0.0;
        poly[deg + 2] = 0.0;
        for( k = deg + 2; k >= 1; k-- )
            poly[k] += b1 * poly[k - 1] + ( k >= 2 ? b2 * poly[k - 2] : 0.0 );
        deg += 2;
    }
    for( i = 0; i < nreal; i++ )
    {
        poly[deg + 1] = 0.0;
        for( k = deg + 1; k >= 1; k-- )
            poly[k] -= real[i] * poly[k - 1];
        deg += 1;
    }

    // A(z) = 1 - sum a_k z^-k, so a_k is minus the polynomial coefficient
    for( k = 1; k <= deg; k++ )
        a[k - 1] = -poly[k];
}




//-----------------------------------------------------------------------------
// name: formants_from_poles()
// desc: one candidate formant per conjugate pair: frequency and bandwidth
//       straight from the pole, level from the envelope of `a` at that
//       frequency. Then sorted and filtered by sort_formants().
//-----------------------------------------------------------------------------
static void formants_from_poles( const double * theta, const double * rad, int npairs,
                                 const double * a, int order, double fs, FormantSet * out )
{
    Formant cand[512];
    int n = 0;
    for( int i = 0; i < npairs && n < 512; i++ )
    {
        if( !( rad[i] > 0.0 && rad[i] < 1.0 ) ) continue;
        cand[n].freq = theta[i] * fs / ( 2.0 * LPC__PI );
        cand[n].bw = -log( rad[i] ) * fs / LPC__PI;
        cand[n].level = 10.0 * log10( envelope_power( a, order, theta[i] ) );
        n++;
    }
    sort_formants( cand, n, out );
}




//-----------------------------------------------------------------------------
// name: lpc_formants_poles()
// desc: measures formants as the poles of the LPC filter: each conjugate
//       pole pair within the formant range and narrower than FORMANT_MAX_BW
//       is a formant. Unlike envelope peak picking this sees formants that
//       are too close together to make separate peaks, and gives the
//       resonances' true frequencies and bandwidths. On the other hand,
//       an LPC order higher than the voice needs adds poles that don't
//       correspond to real resonances; the bandwidth limit screens out
//       most of them.
//-----------------------------------------------------------------------------
void lpc_formants_poles( const float * coefs, int order, float srate, FormantSet * out )
{
    double a[1024], theta[512], rad[512], real[1024];
    Cx z[1024];
    int npairs, nreal, i;

    out->count = 0;
    if( order < 1 || order > 1023 ) return;
    for( i = 0; i < order; i++ ) a[i] = coefs[i];

    if( !lpc_poles( a, order, z ) ) return;
    if( !split_poles( z, order, theta, rad, &npairs, real, &nreal ) ) return;
    formants_from_poles( theta, rad, npairs, a, order, srate, out );
}




//-----------------------------------------------------------------------------
// name: warp_formants_poles()
// desc: formant shift by moving the formant poles: every conjugate pair
//       that counts as a formant (is_formant()) has its angle multiplied by
//       alpha (capped at POLE_MAX_ANGLE), which multiplies its frequency by
//       alpha, and its radius raised to the power bw_factor, which
//       multiplies its bandwidth by bw_factor:
//         bw_factor = 1      bandwidths stay the same in Hz (the default;
//                            the envelope and LSF warps can't do this)
//         bw_factor = alpha  bandwidths scale with the frequencies, like
//                            the other two warps
//       All other poles (broad ones shaping the spectral slope, real ones)
//       stay where they are. Because the formants are handled one by one,
//       this is also the place to give each formant its own factor.
//       Then finish_warp() restores the original spectral balance, refits
//       the coefficients, and matches the loudness.
//-----------------------------------------------------------------------------
void warp_formants_poles( float * coefs, int order, float srate, float alpha, float * power,
                          FormantSet * before, FormantSet * after, float bw_factor )
{
    const double fs = srate;
    double a[1024], a_edit[1024], theta[512], rad[512], real[1024];
    Cx z[1024];
    int npairs, nreal, i;

    if( before ) before->count = 0;
    if( after ) after->count = 0;
    if( order < 1 || order > 1023 ) return;
    for( i = 0; i < order; i++ ) a[i] = coefs[i];

    if( !lpc_poles( a, order, z ) ) return;
    if( !split_poles( z, order, theta, rad, &npairs, real, &nreal ) ) return;

    // formants as analyzed. "after" starts out the same, in case we bail
    // out below and leave the coefficients as they are.
    FormantSet orig;
    formants_from_poles( theta, rad, npairs, a, order, fs, &orig );
    if( before ) *before = orig;
    if( after ) *after = orig;

    // move the formant poles
    for( i = 0; i < npairs; i++ )
    {
        if( !( rad[i] > 0.0 && rad[i] < 1.0 ) ) continue;
        double freq = theta[i] * fs / ( 2.0 * LPC__PI );
        double bw = -log( rad[i] ) * fs / LPC__PI;
        if( !is_formant( freq, bw ) ) continue;

        theta[i] *= alpha;
        if( theta[i] > POLE_MAX_ANGLE ) theta[i] = POLE_MAX_ANGLE;
        rad[i] = pow( rad[i], (double)bw_factor );
        if( rad[i] > POLE_MAX_RADIUS ) rad[i] = POLE_MAX_RADIUS;
    }

    // rebuild, restore the spectral balance, refit, match loudness
    poles_to_lpc( theta, rad, npairs, real, nreal, a_edit );
    if( !finish_warp( a, a_edit, order, POLE_BALANCE_TERMS, coefs, power ) ) return;

    // measure what will actually be synthesized
    if( after )
        lpc_formants_poles( coefs, order, srate, after );
}




//-----------------------------------------------------------------------------
// name: cos_series_zeros()
// desc: finds the zeros in (0, pi) of
//           G(w) = g[m] + 2 * sum_{k=1}^{m} g[m-k] cos(k w)
//       which is a symmetric polynomial g[0..2m] (g[k] = g[2m-k]) evaluated
//       on the unit circle, times e^(jmw). Scans LSF_GRID intervals for
//       sign changes and refines each by bisection. Writes them to out in
//       increasing order and returns how many were found (m if all went
//       well).
//-----------------------------------------------------------------------------
static double cos_series( const double * g, int m, double w )
{
    // cos(kw) by the recurrence cos((k+1)w) = 2 cos(w) cos(kw) - cos((k-1)w)
    double x = cos( w ), c_prev = 1.0, c = x, sum = g[m];
    for( int k = 1; k <= m; k++ )
    {
        sum += 2.0 * g[m - k] * c;
        double c_next = 2.0 * x * c - c_prev;
        c_prev = c;
        c = c_next;
    }
    return sum;
}

static int cos_series_zeros( const double * g, int m, double * out )
{
    int n = 0;
    double w0 = 0.0, v0 = cos_series( g, m, 0.0 );
    for( int j = 1; j <= LSF_GRID && n < m; j++ )
    {
        double w1 = LPC__PI * j / LSF_GRID;
        double v1 = cos_series( g, m, w1 );
        if( ( v0 < 0.0 && v1 >= 0.0 ) || ( v0 > 0.0 && v1 <= 0.0 ) )
        {
            // bisect [w0, w1] down to ~1e-13 rad
            double lo = w0, hi = w1, vlo = v0;
            for( int it = 0; it < 45; it++ )
            {
                double mid = 0.5 * ( lo + hi ), vm = cos_series( g, m, mid );
                if( ( vlo < 0.0 ) == ( vm < 0.0 ) ) { lo = mid; vlo = vm; }
                else hi = mid;
            }
            double root = 0.5 * ( lo + hi );
            if( root > 0.0 && root < LPC__PI ) out[n++] = root;
        }
        w0 = w1;
        v0 = v1;
    }
    return n;
}




//-----------------------------------------------------------------------------
// name: lpc_to_lsf()
// desc: computes the line spectral frequencies of A(z), in radians and in
//       increasing order, into lsf[0..order-1]; which[i] says whether
//       lsf[i] is a root of P (0) or of Q (1). With A_k the coefficients of
//       A(z) (A_0 = 1, A_k = -a[k-1], A_(p+1) = 0):
//           P(z) = A(z) + z^-(p+1) A(1/z)    P_k = A_k + A_(p+1-k)
//           Q(z) = A(z) - z^-(p+1) A(1/z)    Q_k = A_k - A_(p+1-k)
//       P and Q also have trivial roots at z = -1 and/or z = 1, divided
//       out first:
//           even p:  P / (1 + z^-1),  Q / (1 - z^-1)
//           odd p:   P as is,         Q / (1 - z^-2)
//       What's left are symmetric polynomials of even degree whose roots
//       are conjugate pairs on the unit circle, found with
//       cos_series_zeros(). Returns false unless exactly `order` strictly
//       increasing LSFs were found (always the case for a stable filter).
//-----------------------------------------------------------------------------
static bool lpc_to_lsf( const double * a, int order, double * lsf, int * which )
{
    const int p = order;
    double A[1026], P[1026], Q[1026], Pr[1026], Qr[1026];
    double zp[512], zq[512];
    int k, degP, degQ;

    if( order < 1 || order > 1023 ) return false;
    A[0] = 1.0;
    for( k = 1; k <= p; k++ ) A[k] = -a[k - 1];
    A[p + 1] = 0.0;
    for( k = 0; k <= p + 1; k++ )
    {
        P[k] = A[k] + A[p + 1 - k];
        Q[k] = A[k] - A[p + 1 - k];
    }

    // divide out the trivial roots (synthetic division)
    if( p % 2 == 0 )
    {
        Pr[0] = P[0];
        Qr[0] = Q[0];
        for( k = 1; k <= p; k++ )
        {
            Pr[k] = P[k] - Pr[k - 1];       // / (1 + z^-1)
            Qr[k] = Q[k] + Qr[k - 1];       // / (1 - z^-1)
        }
        degP = degQ = p;
    }
    else
    {
        for( k = 0; k <= p + 1; k++ ) Pr[k] = P[k];
        for( k = 0; k <= p - 1; k++ )       // / (1 - z^-2)
            Qr[k] = Q[k] + ( k >= 2 ? Qr[k - 2] : 0.0 );
        degP = p + 1;
        degQ = p - 1;
    }

    int nP = cos_series_zeros( Pr, degP / 2, zp );
    int nQ = cos_series_zeros( Qr, degQ / 2, zq );
    if( nP != degP / 2 || nQ != degQ / 2 ) return false;

    // merge the two sorted lists
    int i = 0, j = 0, n = 0;
    while( i < nP || j < nQ )
    {
        if( j >= nQ || ( i < nP && zp[i] < zq[j] ) ) { lsf[n] = zp[i++]; which[n++] = 0; }
        else { lsf[n] = zq[j++]; which[n++] = 1; }
    }
    for( k = 1; k < p; k++ )
        if( !( lsf[k] > lsf[k - 1] ) ) return false;
    return true;
}




//-----------------------------------------------------------------------------
// name: lsf_to_lpc()
// desc: the inverse of lpc_to_lsf(): rebuilds P and Q as products of
//       (1 - 2 cos(w) z^-1 + z^-2) over their LSFs, puts the trivial roots
//       back, and takes A = (P + Q) / 2. which[] must be what lpc_to_lsf()
//       gave for these LSFs (editing their values doesn't change it), and
//       lsf[] must be in increasing order. Each product is multiplied in
//       spread_order() to keep precision.
//-----------------------------------------------------------------------------
static void lsf_to_lpc( const double * lsf, const int * which, int order, double * a )
{
    const int p = order;
    double P[1027], Q[1027];
    double wP[512], wQ[512];
    int spread[512];
    int nP = 0, nQ = 0, degP = 0, degQ = 0, i, k, n;
    P[0] = Q[0] = 1.0;

    // split into P's and Q's LSFs (each list stays in increasing order)
    for( i = 0; i < p; i++ )
    {
        if( which[i] ) wQ[nQ++] = lsf[i];
        else wP[nP++] = lsf[i];
    }

    for( int pass = 0; pass < 2; pass++ )
    {
        double * poly = pass ? Q : P;
        const double * w = pass ? wQ : wP;
        int & deg = pass ? degQ : degP;
        int count = pass ? nQ : nP;

        spread_order( count, spread );
        for( n = 0; n < count; n++ )
        {
            double b1 = -2.0 * cos( w[spread[n]] );
            poly[deg + 1] = 0.0;
            poly[deg + 2] = 0.0;
            for( k = deg + 2; k >= 1; k-- )
                poly[k] += b1 * poly[k - 1] + ( k >= 2 ? poly[k - 2] : 0.0 );
            deg += 2;
        }
    }

    if( p % 2 == 0 )
    {
        // P *= (1 + z^-1), Q *= (1 - z^-1)
        P[degP + 1] = 0.0;
        Q[degQ + 1] = 0.0;
        for( k = degP + 1; k >= 1; k-- ) P[k] += P[k - 1];
        for( k = degQ + 1; k >= 1; k-- ) Q[k] -= Q[k - 1];
    }
    else
    {
        // Q *= (1 - z^-2)
        Q[degQ + 1] = 0.0;
        Q[degQ + 2] = 0.0;
        for( k = degQ + 2; k >= 2; k-- ) Q[k] -= Q[k - 2];
    }

    for( k = 1; k <= p; k++ )
        a[k - 1] = -0.5 * ( P[k] + Q[k] );
}




//-----------------------------------------------------------------------------
// name: formants_from_lsf()
// desc: estimates formants from LSFs. Near a sharp resonance the phase of
//       A(z) swings quickly, which forces one root of P and one of Q close
//       together around the resonance. So each pair of neighboring LSFs
//       whose spacing is a local minimum (narrower than the pairs on either
//       side) is taken as a formant:
//           frequency  the midpoint of the pair
//           bandwidth  about the spacing of the pair. For an isolated
//                      resonance the spacing equals the bandwidth when the
//                      pair sits symmetrically around it, and is wider
//                      otherwise, so this tends to overestimate.
//       This is a rough estimate; the pole method is the accurate one.
//-----------------------------------------------------------------------------
static void formants_from_lsf( const double * lsf, int order,
                               const double * a, double fs, FormantSet * out )
{
    Formant cand[512];
    int n = 0;
    for( int i = 0; i + 1 < order && n < 512; i++ )
    {
        double d = lsf[i + 1] - lsf[i];
        bool narrower_left = ( i == 0 ) || d < lsf[i] - lsf[i - 1];
        bool narrower_right = ( i + 2 >= order ) || d < lsf[i + 2] - lsf[i + 1];
        if( !( narrower_left && narrower_right ) ) continue;

        double mid = 0.5 * ( lsf[i] + lsf[i + 1] );
        cand[n].freq = mid * fs / ( 2.0 * LPC__PI );
        cand[n].bw = d * fs / ( 2.0 * LPC__PI );
        cand[n].level = 10.0 * log10( envelope_power( a, order, mid ) );
        n++;
    }
    sort_formants( cand, n, out );
}




//-----------------------------------------------------------------------------
// name: lpc_formants_lsf()
// desc: measures formants from the LPC filter's line spectral frequencies
//       (see formants_from_lsf()). Cheaper than the pole method and never
//       fails on a stable filter, but the bandwidths are only estimates.
//-----------------------------------------------------------------------------
void lpc_formants_lsf( const float * coefs, int order, float srate, FormantSet * out )
{
    double a[1024], lsf[1024];
    int which[1024], i;

    out->count = 0;
    if( order < 1 || order > 1023 ) return;
    for( i = 0; i < order; i++ ) a[i] = coefs[i];

    if( !lpc_to_lsf( a, order, lsf, which ) ) return;
    formants_from_lsf( lsf, order, a, srate, out );
}




//-----------------------------------------------------------------------------
// name: warp_formants_lsf()
// desc: formant shift on line spectral frequencies: every LSF is mapped
//       through warp_angle(), which moves each close pair (each formant)
//       to alpha times its frequency. Pair spacings scale too, so bandwidths
//       scale with alpha, as with the envelope warp. warp_angle() keeps the
//       LSFs in order inside (0, pi), which is exactly the condition for a
//       stable filter; as a safety net they're also kept at least
//       LSF_MIN_GAP_HZ apart. Then finish_warp() restores the original
//       spectral balance, refits the coefficients, and matches the loudness.
//-----------------------------------------------------------------------------
void warp_formants_lsf( float * coefs, int order, float srate, float alpha, float * power,
                        FormantSet * before, FormantSet * after )
{
    double a[1024], a_edit[1024], lsf[1024];
    int which[1024], i;

    if( before ) before->count = 0;
    if( after ) after->count = 0;
    if( order < 1 || order > 1023 ) return;
    for( i = 0; i < order; i++ ) a[i] = coefs[i];

    if( !lpc_to_lsf( a, order, lsf, which ) ) return;

    // formants as analyzed ("after" too, in case we bail out below)
    FormantSet orig;
    formants_from_lsf( lsf, order, a, srate, &orig );
    if( before ) *before = orig;
    if( after ) *after = orig;

    // warp
    for( i = 0; i < order; i++ )
        lsf[i] = warp_angle( lsf[i], alpha );

    // safety net: keep them in (0, pi), increasing, at least `gap` apart:
    // first cap them from the top down, then push them up from the bottom.
    // (gap is small enough that `order` of them always fit.)
    const double gap = 2.0 * LPC__PI * LSF_MIN_GAP_HZ / srate;
    double hi = LPC__PI - gap;
    for( i = order - 1; i >= 0; i-- )
    {
        if( lsf[i] > hi ) lsf[i] = hi;
        hi = lsf[i] - gap;
    }
    double lo = gap;
    for( i = 0; i < order; i++ )
    {
        if( lsf[i] < lo ) lsf[i] = lo;
        lo = lsf[i] + gap;
    }

    // rebuild, restore the spectral balance, refit, match loudness
    lsf_to_lpc( lsf, which, order, a_edit );
    if( !finish_warp( a, a_edit, order, LSF_BALANCE_TERMS, coefs, power ) ) return;

    // measure what will actually be synthesized
    if( after )
        lpc_formants_lsf( coefs, order, srate, after );
}




//-----------------------------------------------------------------------------
// name: lpc_measure_formants()
// desc: measures the resonances of a frame of speech x[0..n-1] (n <= 1024)
//       at sample rate fs, for reporting rather than resynthesis, with the
//       usual formant-tracking recipe: pre-emphasis, resample to about
//       12 kHz, Hann window, order-14 autocorrelation LPC, and the
//       frequencies and bandwidths of its poles (those above 90 Hz and up
//       to 100 Hz below the new Nyquist, narrower than 700 Hz). Writes up
//       to max of them to freq[] and bw[] (Hz), lowest first, and returns
//       how many. Which is F1, F2, F3 is left to the caller. Not
//       thread-safe (static work buffers).
//-----------------------------------------------------------------------------
#define MEASURE_FORMANT_FS  12000.0   // resample to about this
#define MEASURE_LPC_ORDER   14

int lpc_measure_formants( const SAMPLE * x, int n, double fs, double * freq, double * bw, int max )
{
    // resample by an integer factor to about MEASURE_FORMANT_FS, low-passing
    // first with a windowed sinc (cutoff at 90% of the new Nyquist)
    int dec = (int)( fs / MEASURE_FORMANT_FS + 0.5 );
    if( dec < 1 ) dec = 1;
    const double fs2 = fs / dec;
    static double y[1024], h[64];                 // static: not thread-safe
    const int taps = dec > 1 ? 31 : 1;
    if( dec > 1 )
    {
        const double fc = 0.9 / dec;              // cutoff, relative to the input Nyquist
        double sum = 0.0;
        for( int k = 0; k < taps; k++ )
        {
            double m = k - ( taps - 1 ) / 2.0;
            double sinc = m == 0.0 ? fc : sin( LPC__PI * fc * m ) / ( LPC__PI * m );
            h[k] = sinc * ( 0.54 - 0.46 * cos( 2.0 * LPC__PI * k / ( taps - 1 ) ) );
            sum += h[k];
        }
        for( int k = 0; k < taps; k++ ) h[k] /= sum;
    }
    else h[0] = 1.0;

    int m = 0;
    for( int i = taps - 1; i < n && m < 1024; i += dec )
    {
        double s = 0.0;
        for( int k = 0; k < taps; k++ ) s += h[k] * x[i - k];
        y[m++] = s;
    }
    if( m < 4 * MEASURE_LPC_ORDER ) return 0;

    // pre-emphasis (tilts the spectrum up so the higher formants get modeled)
    // and a Hann window
    double prev = y[0];
    for( int i = 0; i < m; i++ )
    {
        double cur = y[i];
        y[i] = ( cur - 0.97 * prev ) * ( 0.5 - 0.5 * cos( 2.0 * LPC__PI * i / ( m - 1 ) ) );
        prev = cur;
    }

    // autocorrelation and Levinson-Durbin
    const int p = MEASURE_LPC_ORDER;
    double r[MEASURE_LPC_ORDER + 1], a[MEASURE_LPC_ORDER + 1], tmp[MEASURE_LPC_ORDER + 1];
    for( int k = 0; k <= p; k++ )
    {
        double s = 0.0;
        for( int i = k; i < m; i++ ) s += y[i] * y[i - k];
        r[k] = s;
    }
    if( !( r[0] > 0.0 ) ) return 0;
    r[0] *= 1.0 + 1e-9;                           // a whisker of white noise for stability
    double E = r[0];
    for( int i = 1; i <= p; i++ )
    {
        double acc = r[i];
        for( int j = 1; j < i; j++ ) acc -= a[j] * r[i - j];
        double k = acc / E;
        if( !( fabs( k ) < 1.0 ) ) return 0;
        tmp[i] = k;
        for( int j = 1; j < i; j++ ) tmp[j] = a[j] - k * a[i - j];
        for( int j = 1; j <= i; j++ ) a[j] = tmp[j];
        E *= 1.0 - k * k;
    }

    // poles -> resonances, lowest first
    Cx z[MEASURE_LPC_ORDER];
    if( !lpc_poles( a + 1, p, z ) ) return 0;
    double f_all[MEASURE_LPC_ORDER], b_all[MEASURE_LPC_ORDER];
    int count = 0;
    for( int i = 0; i < p; i++ )
    {
        if( z[i].im <= 1e-8 ) continue;           // one of each conjugate pair; skip real poles
        double rad = sqrt( z[i].re * z[i].re + z[i].im * z[i].im );
        if( !( rad > 0.0 && rad < 1.0 ) ) continue;
        double f = atan2( z[i].im, z[i].re ) * fs2 / ( 2.0 * LPC__PI );
        double b = -log( rad ) * fs2 / LPC__PI;
        if( f < 90.0 || f > fs2 / 2.0 - 100.0 || b > 700.0 ) continue;
        int j = count++;                           // insertion sort by frequency
        while( j > 0 && f_all[j - 1] > f ) { f_all[j] = f_all[j - 1]; b_all[j] = b_all[j - 1]; j--; }
        f_all[j] = f;
        b_all[j] = b;
    }
    if( count > max ) count = max;
    for( int i = 0; i < count; i++ ) { freq[i] = f_all[i]; bw[i] = b_all[i]; }
    return count;
}
