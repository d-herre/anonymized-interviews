//-----------------------------------------------------------------------------
// name: psola.h
// desc: streaming TD-PSOLA pitch shifter, guided by the LPC residual.
//
//       Changes the pitch of a voice without changing its duration or its
//       formants. rt_lpc uses it for the residual excitation: the voice is
//       pitch-shifted here, then whitened with its own (analysis) LPC filter
//       and recolored with the formant-shifted one, both taken at the time
//       each output sample is played.
//
//       How (time-domain pitch-synchronous overlap-add):
//         1. pitch marks: one per pitch period, on the biggest peak of the
//            LPC residual (the "guide") in that period: the glottal
//            closure, where each period's pulse starts. Evenly spaced
//            (every 128 samples) where the frame is unvoiced.
//         2. synthesis marks: spaced like the pitch marks, times the period
//            ratio (output period / input period), so the natural
//            period-to-period variation is kept. For each one, the SIGNAL
//            around the nearest pitch mark (back toward the mark before it,
//            on toward the mark after) is cut out with a Hann window - a
//            "grain", one period of the voice - and added in there. Closer
//            spacing = higher pitch; periods are repeated or skipped as
//            needed to keep the timing. A grain never reaches past the
//            neighboring synthesis marks, so at most two overlap.
//         Unvoiced stretches use a ratio of 1, so noise goes through
//         unchanged, and with a pitch factor of 1 the signal comes out
//         exactly as it went in (just delayed).
//
//       Each grain keeps the vocal-tract ringing of its own period, which
//       is what keeps the formants where they were.
//
//       Streaming: push() a block of signal and guide as they come, then
//       pull() a block of output. The output runs latency() samples behind
//       what was pushed: a grain's falling half has to wait until the next
//       pitch mark is found, which takes up to 1.25 periods after the
//       grain's own mark. With a latency of 512 samples at 24 kHz that's
//       always in time for voices down to about 60 Hz; below that the gap
//       to the next mark is guessed from the period.
//-----------------------------------------------------------------------------
#ifndef __PSOLA_H__
#define __PSOLA_H__

#include <deque>
#include <vector>
#include <cstdint>

class PsolaShifter
{
public:
    // block:      samples per push()/pull(); also the latency
    // max_period: longest pitch period the analysis can report, in samples
    PsolaShifter( int block, int max_period );

    // the next n samples (n <= block) of the signal to shift and of its
    // LPC residual (the guide for the pitch marks). period is the pitch
    // period measured for them, in samples (0 = unvoiced); out_period is
    // the period wanted in the output (<= 0 = same as period).
    void push( const float * signal, const float * guide, int n,
               float period, float out_period );

    // the next n samples of pitch-shifted signal, latency() samples behind
    // what was pushed
    void pull( float * out, int n );

    int latency() const { return m_latency; }

    // forget everything (e.g. after the stream restarts)
    void reset();

private:
    struct Mark
    {
        int64_t pos;        // where, on the residual's timeline
        float period;       // pitch period measured there (samples)
        float ratio;        // output period / input period (1 if unvoiced)
        int prev_gap;       // samples since the previous mark (or the period)
        int next_gap;       // samples to the next mark; -1 until it's placed
        bool voiced;
    };

    float & res( int64_t t ) { return m_res[ (size_t)t & m_mask ]; }
    float & sig( int64_t t ) { return m_sig[ (size_t)t & m_mask ]; }
    float & acc( int64_t t ) { return m_acc[ (size_t)t & m_mask ]; }
    void place_marks( float period, float out_period );
    const Mark * nearest_usable( double t ) const;
    void add_half( const Mark & a, int64_t ts, int from, int to, int half );

    int m_latency;
    int m_max_period;
    int m_unvoiced;          // pitch-mark spacing in unvoiced stretches
    size_t m_mask;           // ring size - 1
    std::vector<float> m_res;  // guide (residual) ring buffer
    std::vector<float> m_sig;  // signal ring buffer
    std::vector<float> m_acc;  // output (overlap-add) ring buffer
    int64_t m_written;       // residual samples pushed so far
    int64_t m_read;          // next output time pull() returns
    int64_t m_last_mark;     // last pitch mark placed (-1 = none yet)
    bool m_last_voiced;
    double m_next_synth;     // next synthesis mark
    double m_prev_synth;     // the last one (NAN: none to go by)
    int m_prev_kind;         // the last grain: -1 none, 0 unvoiced, 1 voiced
    // a grain whose rising half is in, waiting to learn where the next pitch
    // mark is before adding its falling half (and placing the next
    // synthesis mark)
    bool m_pending;
    Mark m_pending_mark;
    std::deque<Mark> m_marks;
};

#endif
