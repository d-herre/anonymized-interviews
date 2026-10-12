//-----------------------------------------------------------------------------
// name: psola.cpp
// desc: streaming TD-PSOLA pitch shifter, guided by the LPC residual
//       (see psola.h)
//-----------------------------------------------------------------------------
#include "psola.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// how far a pitch mark may move from one period after the last one, as a
// fraction of the period: allows for jitter and gliding pitch, without
// letting the marks lock onto something else between pulses
#define MARK_SEARCH_LO   0.75
#define MARK_SEARCH_HI   1.25




//-----------------------------------------------------------------------------
// name: PsolaShifter()
// desc: block: samples per push()/pull() (and the latency); max_period: the
//       longest pitch period that will be reported, in samples
//-----------------------------------------------------------------------------
PsolaShifter::PsolaShifter( int block, int max_period )
{
    m_latency = block;
    m_max_period = std::max( max_period, 2 );
    // in unvoiced stretches: marks (and grains) every ~5 ms at 24 kHz. Any
    // spacing works there, since nothing is shifted: the grains are put
    // back with the same spacing and their Hann windows add up to 1.
    m_unvoiced = 128;

    // the ring buffers must hold the latency, a block, and a grain on each
    // side of a mark, with room to spare
    size_t need = (size_t)( 4 * ( block + 2 * m_max_period ) );
    size_t size = 1;
    while( size < need ) size <<= 1;
    m_mask = size - 1;
    m_res.assign( size, 0.0f );
    m_sig.assign( size, 0.0f );
    m_acc.assign( size, 0.0f );
    reset();
}




//-----------------------------------------------------------------------------
// name: reset()
// desc: back to the starting state: silence, no pitch marks
//-----------------------------------------------------------------------------
void PsolaShifter::reset()
{
    std::fill( m_res.begin(), m_res.end(), 0.0f );
    std::fill( m_sig.begin(), m_sig.end(), 0.0f );
    std::fill( m_acc.begin(), m_acc.end(), 0.0f );
    m_written = 0;
    m_read = -m_latency;    // the output starts latency samples "early"
    m_last_mark = -1;
    m_last_voiced = false;
    m_next_synth = 0;
    m_prev_synth = NAN;
    m_prev_kind = -1;
    m_pending = false;
    m_marks.clear();
}




//-----------------------------------------------------------------------------
// name: push()
// desc: appends n samples of signal and guide, then places every pitch
//       mark the new samples allow
//-----------------------------------------------------------------------------
void PsolaShifter::push( const float * signal, const float * guide, int n,
                         float period, float out_period )
{
    for( int i = 0; i < n; i++ )
    {
        sig( m_written + i ) = signal[i];
        res( m_written + i ) = guide[i];
    }
    m_written += n;

    // a period the analysis can't report means a measurement problem: treat
    // it as unvoiced rather than search far outside the buffers
    if( !( period >= 2 && period <= m_max_period ) ) period = 0;
    if( !( out_period > 0 ) ) out_period = period;
    // pitch factors far above 1 can ask for absurdly short periods
    out_period = std::max( out_period, 2.0f );
    place_marks( period, out_period );
}




//-----------------------------------------------------------------------------
// name: place_marks()
// desc: pitch marks up to where the guide (residual) has been pushed.
//       voiced:   the biggest residual peak in a range one period (+/- 25%)
//                 after the last mark. Coming out of an unvoiced stretch (or
//                 at the start) there's no last pulse to go by, so it looks
//                 at the whole next period instead.
//       unvoiced: every m_unvoiced samples
//       All the marks placed by one call get that call's periods.
//-----------------------------------------------------------------------------
void PsolaShifter::place_marks( float period, float out_period )
{
    while( true )
    {
        Mark mark;
        const int64_t last = m_last_mark;
        if( period > 0 )
        {
            int lo, hi;
            if( m_last_voiced )
            {
                lo = (int)lrint( MARK_SEARCH_LO * period );
                hi = (int)lrint( MARK_SEARCH_HI * period );
            }
            else
            {
                lo = 1;
                hi = (int)lrint( period );
            }
            lo = std::max( lo, 1 );
            hi = std::max( hi, lo );
            // the whole range must have arrived
            if( last + hi >= m_written ) break;

            int64_t best = last + lo;
            float best_v = -1.0f;
            for( int64_t t = last + lo; t <= last + hi; t++ )
            {
                float v = fabsf( res( t ) );
                if( v > best_v ) { best_v = v; best = t; }
            }
            mark.pos = best;
            mark.period = period;
            mark.ratio = out_period / period;
            mark.voiced = true;
        }
        else
        {
            if( last + m_unvoiced >= m_written ) break;
            mark.pos = last + m_unvoiced;
            mark.period = (float)m_unvoiced;
            mark.ratio = 1.0f;
            mark.voiced = false;
        }

        // the gaps to the neighbors (even between a voiced and an unvoiced
        // mark: then the grains on either side of the gap still fade into
        // each other exactly)
        mark.next_gap = -1;
        mark.prev_gap = (int)lrintf( mark.period );
        if( !m_marks.empty() )
            m_marks.back().next_gap = mark.prev_gap = (int)( mark.pos - m_marks.back().pos );

        m_marks.push_back( mark );
        m_last_mark = mark.pos;
        m_last_voiced = mark.voiced;
    }

    // safety: never let the list grow without bound (pull() normally trims it)
    while( m_marks.size() > 4096 ) m_marks.pop_front();
}




//-----------------------------------------------------------------------------
// name: next_gap_of()
// desc: a mark's gap to the next mark, or its period while that isn't
//       known yet
//-----------------------------------------------------------------------------
static inline int next_gap_of( int next_gap, float period )
{
    return next_gap > 0 ? next_gap : std::max( 1, (int)lrintf( period ) );
}




//-----------------------------------------------------------------------------
// name: nearest_usable()
// desc: the pitch mark closest to time t that a grain can be cut around,
//       or NULL if there's none
//-----------------------------------------------------------------------------
const PsolaShifter::Mark * PsolaShifter::nearest_usable( double t ) const
{
    const Mark * best = NULL;
    double best_d = 0;
    const int64_t oldest = m_written - (int64_t)m_mask;   // oldest sample still in the ring
    for( const Mark & m : m_marks )
    {
        double d = fabs( (double)m.pos - t );
        if( best && (double)m.pos > t && d >= best_d ) break;   // only getting farther
        // its rising half must still be in the ring (the falling half
        // comes later, and add_half() stops at the end of the signal)
        if( m.pos >= m_written || m.pos - m.prev_gap < oldest ) continue;
        if( !best || d < best_d ) { best = &m; best_d = d; }
    }
    return best;
}




//-----------------------------------------------------------------------------
// name: add_half()
// desc: overlap-adds samples from..to-1 (relative to the marks) of a grain:
//       the signal around pitch mark a, under half a Hann window `half`
//       samples long (rising for negative offsets, falling for positive),
//       placed around synthesis mark ts. Skips anything already handed out.
//       Raising the pitch (ratio < 1) needs no scaling: the grains are cut
//       to the closer spacing, so their windows still add up to 1. Lowering
//       it leaves dips between grains (each holds only one period), so
//       sqrt(ratio) makes up the lost power.
//-----------------------------------------------------------------------------
void PsolaShifter::add_half( const Mark & a, int64_t ts, int from, int to, int half )
{
    const float scale = a.ratio > 1.0f ? sqrtf( a.ratio ) : 1.0f;
    const float k = (float)M_PI / half;
    // (a falling half can only be cut short by an unusually low voice,
    // when it has to go in before the signal after its mark has arrived)
    to = (int)std::min( (int64_t)to, m_written - a.pos );
    for( int i = from; i < to; i++ )
    {
        const int64_t t = ts + i;
        if( t < m_read ) continue;
        acc( t ) += scale * ( .5f + .5f * cosf( k * i ) ) * sig( a.pos + i );
    }
}




//-----------------------------------------------------------------------------
// name: pull()
// desc: n samples of output. First adds every grain that reaches into
//       them, one at each synthesis mark, then hands them over.
//       Each grain goes in in two halves. The rising half reaches back
//       toward the previous synthesis mark. The falling half reaches on
//       toward the next synthesis mark, which is the gap to the next PITCH
//       mark (times the ratio) further on, so it waits until that pitch
//       mark has been found - unless the output has caught up with it.
//-----------------------------------------------------------------------------
void PsolaShifter::pull( float * out, int n )
{
    const int64_t end = m_read + n;

    // after a stretch with nothing usable, don't try to fill in the past
    if( !m_pending && m_next_synth < m_read )
    {
        m_next_synth = (double)m_read;
        m_prev_synth = NAN;
    }

    while( true )
    {
        const int64_t ts = (int64_t)llround( m_next_synth );

        if( m_pending )
        {
            // the grain's falling half. Look the mark up again: its gap to
            // the next mark may be known by now.
            Mark a = m_pending_mark;
            for( const Mark & m : m_marks )
                if( m.pos == a.pos ) { a = m; break; }
            if( a.next_gap <= 0 && ts >= end ) break;   // can wait for it
            const double right_s = next_gap_of( a.next_gap, a.period ) * (double)a.ratio;
            // no further than the next pitch mark (one pulse per grain) or
            // the next synthesis mark (at most two grains overlap)
            const int hr = std::max( 1, (int)lrint( std::min( (double)next_gap_of( a.next_gap, a.period ), right_s ) ) );
            add_half( a, ts, 0, hr, hr );
            m_pending = false;
            m_prev_synth = m_next_synth;
            m_prev_kind = (int)a.voiced;
            m_next_synth += std::max( right_s, 1.0 );
            continue;
        }

        const Mark * a = nearest_usable( m_next_synth );
        if( !a ) break;                 // nothing to work with yet: silence

        // where voicing starts or stops, put the synthesis mark right on the
        // pitch mark, so the two grids line up again (with a ratio of 1 they
        // then stay lined up, and the signal passes through unchanged).
        // Only ever forward, past the last grain: going back would put
        // grains in twice.
        if( (int)a->voiced != m_prev_kind && a->pos >= m_read &&
            ( std::isnan( m_prev_synth ) || (double)a->pos > m_prev_synth ) )
        {
            m_next_synth = (double)a->pos;
            m_prev_synth = NAN;
        }
        const int64_t t0 = (int64_t)llround( m_next_synth );

        // the rising half reaches back to the previous synthesis mark, but
        // no further than the previous pitch mark
        const double left_s = std::isnan( m_prev_synth ) ? a->prev_gap * (double)a->ratio
                                                         : m_next_synth - m_prev_synth;
        const int hl = std::max( 1, (int)lrint( std::min( (double)a->prev_gap, left_s ) ) );
        // a grain that starts after this output block waits for the next
        // pull(), when more pitch marks may be known
        if( t0 - hl >= end ) break;

        add_half( *a, t0, -hl + 1, 0, hl );
        m_pending = true;
        m_pending_mark = *a;
    }

    // marks before the next synthesis mark, but one, can't be the nearest
    // any more (and the pending grain's mark has been copied)
    while( m_marks.size() >= 2 && (double)m_marks[1].pos <= m_next_synth &&
           m_marks[1].pos + m_max_period <= m_written &&
           !( m_pending && m_marks[0].pos == m_pending_mark.pos ) )
        m_marks.pop_front();

    for( int i = 0; i < n; i++ )
    {
        out[i] = acc( m_read + i );
        acc( m_read + i ) = 0.0f;
    }
    m_read = end;
}
