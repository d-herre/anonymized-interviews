//-----------------------------------------------------------------------------
// name: silk_analysis.cpp
// desc: rt_lpc's SILK analysis (see silk_analysis.h); the SILK code itself
//       is in silk_port.c
//-----------------------------------------------------------------------------
#include "silk_analysis.h"
#include "silk_port.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// SILK's code works on 16-bit-scaled samples
#define SILK_SCALE 32768.0f

static int gcd( int a, int b ) { while( b ) { int t = a % b; a = b; b = t; } return a; }




//-----------------------------------------------------------------------------
// name: SilkAnalysis()
// desc: picks the analysis rate, builds the resampling kernel, and creates
//       the SILK VAD and pitch estimator
//-----------------------------------------------------------------------------
SilkAnalysis::SilkAnalysis( int srate )
{
    m_srate = srate;
    m_fs_khz = srate >= 16000 ? 16 : srate >= 12000 ? 12 : 8;
    m_vad = silk_port_vad_create( m_fs_khz );
    m_pitch = silk_port_pitch_create( m_fs_khz );

    // resampling kernel: windowed sinc, cutoff at 95% of the output Nyquist,
    // 16 zero crossings each side. Output sample m sits at input position
    // m * M / L; its taps are tabulated for each of the L fractional phases.
    const int out = m_fs_khz * 1000, g = gcd( out, srate );
    m_L = out / g;
    m_M = srate / g;
    if( m_L == m_M ) { m_half = 1; m_kernel.assign( 2, 0.0f ); m_kernel[0] = 1.0f; }   // identity
    else
    {
        const double fc = 0.95 * std::min( 1.0, (double)m_L / m_M );    // relative to the input Nyquist
        m_half = (int)ceil( 16.0 / fc );
        m_kernel.assign( (size_t)m_L * 2 * m_half, 0.0f );
        for( int ph = 0; ph < m_L; ph++ )
            for( int j = -m_half + 1; j <= m_half; j++ )
            {
                const double d = j - (double)ph / m_L;     // input sample - output position
                if( fabs( d ) >= m_half ) continue;
                const double t = fc * d;
                const double sinc = fabs( t ) < 1e-12 ? 1.0 : sin( M_PI * t ) / ( M_PI * t );
                m_kernel[(size_t)ph * 2 * m_half + ( j + m_half - 1 )] =
                    (float)( fc * sinc * ( 0.5 + 0.5 * cos( M_PI * d / m_half ) ) );
            }
    }
    m_in_start = 0;
    m_in_total = 0;
    m_out_next = 0;
    m_sig_start = 0;
    m_frame_start = 0;
}

SilkAnalysis::~SilkAnalysis()
{
    silk_port_vad_destroy( m_vad );
    silk_port_pitch_destroy( m_pitch );
}




//-----------------------------------------------------------------------------
// name: push()
// desc: takes in more input, resamples what it can, and analyzes every
//       SILK frame that's complete (including its lookahead)
//-----------------------------------------------------------------------------
void SilkAnalysis::push( const float * x, int n )
{
    m_in.insert( m_in.end(), x, x + n );
    m_in_total += n;
    resample_available();
    analyze_available();
}

void SilkAnalysis::resample_available()
{
    const int taps = 2 * m_half;
    while( true )
    {
        // output sample m_out_next: input position n0 + phase / L
        const long long pos = (long long)m_out_next * m_M;
        const long n0 = (long)( pos / m_L );
        const int ph = (int)( pos % m_L );
        if( n0 + m_half >= m_in_total ) break;          // needs input up to n0 + half

        const float * h = &m_kernel[(size_t)ph * taps];
        double s = 0.0;
        for( int j = -m_half + 1; j <= m_half; j++ )
        {
            const long i = n0 + j;
            if( i < m_in_start ) continue;              // before the stream: silence
            s += m_in[(size_t)( i - m_in_start )] * h[j + m_half - 1];
        }
        m_sig.push_back( (float)s * SILK_SCALE );
        m_out_next++;
    }
    // drop input no future output needs
    const long n0 = (long)( (long long)m_out_next * m_M / m_L );
    const long keep_from = n0 - m_half + 1;
    if( keep_from > m_in_start )
    {
        const long drop = std::min( keep_from - m_in_start, (long)m_in.size() );
        m_in.erase( m_in.begin(), m_in.begin() + drop );
        m_in_start += drop;
    }
}

void SilkAnalysis::analyze_available()
{
    const int frame = SILK_PORT_FRAME_MS * m_fs_khz;
    const int mem = SILK_PORT_LTP_MEM_MS * m_fs_khz;
    const int la = SILK_PORT_LA_PITCH_MS * m_fs_khz;
    const int sub = frame / SILK_PORT_NB_SUBFR;
    const long sig_end = m_sig_start + (long)m_sig.size();
    std::vector<float> buf( (size_t)( mem + frame + la ) );
    std::vector<short> s16( (size_t)frame );

    while( m_frame_start + frame + la <= sig_end )
    {
        // [20 ms history | 20 ms frame | 2 ms lookahead]; silence before the start
        const long b0 = m_frame_start - mem;
        for( int i = 0; i < (int)buf.size(); i++ )
        {
            const long t = b0 + i;
            buf[i] = t < m_sig_start ? 0.0f : m_sig[(size_t)( t - m_sig_start )];
        }
        for( int i = 0; i < frame; i++ )
            s16[i] = (short)std::max( -32768.0f, std::min( 32767.0f, rintf( buf[mem + i] ) ) );

        // SILK's order: VAD on the frame, then the pitch estimator
        int tilt = 0;
        const int activity = silk_port_vad_frame( m_vad, &s16[0], &tilt );
        int lags[SILK_PORT_NB_SUBFR];
        silk_port_pitch_frame( m_pitch, &buf[0], activity, tilt, lags );

        for( int k = 0; k < SILK_PORT_NB_SUBFR; k++ )
        {
            Subframe s;
            s.start = m_frame_start + k * sub;
            s.end = s.start + sub;
            s.lag = lags[k];
            m_subframes.push_back( s );
        }
        m_frame_start += frame;
    }

    // keep the history the next frame needs, and the last ~2 s of results
    const long keep_from = m_frame_start - mem;
    if( keep_from > m_sig_start )
    {
        const long drop = std::min( keep_from - m_sig_start, (long)m_sig.size() );
        m_sig.erase( m_sig.begin(), m_sig.begin() + drop );
        m_sig_start += drop;
    }
    while( m_subframes.size() > 400 ) m_subframes.pop_front();
}




//-----------------------------------------------------------------------------
// name: pitch_over()
// desc: see silk_analysis.h
//-----------------------------------------------------------------------------
float SilkAnalysis::pitch_over( long start, long end ) const
{
    if( m_subframes.empty() ) return 0.0f;
    // the stretch, in analysis-rate samples (doubled, to compare with
    // doubled subframe centers)
    const double scale = (double)m_fs_khz * 1000.0 / m_srate;
    const double a2 = 2.0 * start * scale, b2 = 2.0 * end * scale;

    int count = 0, voiced = 0;
    double lag_sum = 0.0;
    for( const Subframe & s : m_subframes )
    {
        const double c2 = (double)( s.start + s.end );
        if( c2 < a2 || c2 >= b2 ) continue;
        count++;
        if( s.lag > 0 ) { voiced++; lag_sum += s.lag; }
    }
    if( count == 0 )
    {
        // nothing analyzed in the stretch yet: the most recent subframe
        const Subframe & s = m_subframes.back();
        if( (double)( s.start + s.end ) >= b2 ) return 0.0f;    // stretch is older than the results
        count = 1;
        if( s.lag > 0 ) { voiced = 1; lag_sum = s.lag; }
    }
    if( 2 * voiced < count ) return 0.0f;
    return (float)( lag_sum / voiced / scale );
}




//-----------------------------------------------------------------------------
// name: lpc()
// desc: see silk_analysis.h
//-----------------------------------------------------------------------------
void SilkAnalysis::lpc( const float * x, int len, float * coefs, int order )
{
    std::vector<float> scaled( (size_t)len );
    for( int i = 0; i < len; i++ ) scaled[i] = x[i] * SILK_SCALE;
    silk_port_burg( coefs, &scaled[0], len, order, 1.0f / SILK_PORT_MAX_PREDICTION_POWER_GAIN );
}
