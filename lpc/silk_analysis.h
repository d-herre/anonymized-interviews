//-----------------------------------------------------------------------------
// name: silk_analysis.h
// desc: rt_lpc's "silk" analysis: LPC coefficients by Burg's method and
//       pitch from SILK's voice activity detector and pitch estimator (the
//       libopus code in silk_port.c), in place of lpc_analyze()'s
//       autocorrelation fit and autocorrelation pitch detector.
//
//       SILK analyzes speech at 8, 12 or 16 kHz in 20 ms frames, each frame
//       carrying on from the last (the VAD's noise estimates and the pitch
//       estimator's previous lag). So the input is resampled as it arrives
//       and analyzed on SILK's own 20 ms grid, and rt_lpc asks for the pitch
//       of any stretch of input already analyzed: pitch_over().
//
//       One departure from SILK: the pitch estimator searches every frame,
//       not only those its VAD finds voice activity in (see
//       silk_port_pitch_set_vad_gating()). The VAD can take a steady vowel
//       for background noise, especially in its first seconds; SILK would
//       call that frame unvoiced, and with the residual excitation an
//       unvoiced frame isn't pitch-shifted, so the speaker's own pitch would
//       come through.
//-----------------------------------------------------------------------------
#ifndef __SILK_ANALYSIS_H__
#define __SILK_ANALYSIS_H__

#include <vector>
#include <deque>

struct silk_port_vad;
struct silk_port_pitch;

class SilkAnalysis
{
public:
    // srate: the rate of the input pushed in. The analysis runs at 16 kHz
    // (or 12 or 8 kHz for input slower than 16 or 12 kHz).
    SilkAnalysis( int srate );
    ~SilkAnalysis();

    // the next n input samples (+/-1.0, at srate); analyzes every SILK frame
    // they complete
    void push( const float * x, int n );

    // the pitch period, in input samples, over input samples [start, end)
    // (counted from the first one pushed), or 0 if unvoiced: the mean lag
    // of the SILK subframes centered in that stretch, if at least half of
    // them are voiced. Subframes not analyzed yet (the last ~20 ms, which
    // wait for SILK's lookahead) are left out; if none is left, the most
    // recent one counts.
    float pitch_over( long start, long end ) const;

    // LPC coefficients (rt_lpc's predictor convention) of x[0..len-1] by
    // Burg's method, SILK's silk_burg_modified_FLP(), with SILK's limit on
    // the prediction gain (40 dB). order <= 100.
    static void lpc( const float * x, int len, float * coefs, int order );

    int analysis_rate() const { return m_fs_khz * 1000; }

private:
    void resample_available();
    void analyze_available();

    int m_srate, m_fs_khz;
    silk_port_vad * m_vad;
    silk_port_pitch * m_pitch;

    // resampling srate -> m_fs_khz kHz: L/M polyphase windowed sinc
    int m_L, m_M;                       // output rate / input rate = L / M (reduced)
    int m_half;                         // kernel half-width, in input samples
    std::vector<float> m_kernel;        // [phase][2 * m_half] taps
    std::vector<float> m_in;            // input not yet fully used (starts at m_in_start)
    long m_in_start;                    // input index of m_in[0]
    long m_in_total;                    // input samples pushed
    long m_out_next;                    // next output (analysis-rate) sample to make

    // the analysis-rate signal, int16-scaled, from m_sig_start on
    std::vector<float> m_sig;
    long m_sig_start;
    long m_frame_start;                 // start of the next SILK frame to analyze

    // results: one per analyzed subframe, oldest first
    struct Subframe { long start; long end; int lag; };   // analysis-rate samples; lag 0 = unvoiced
    std::deque<Subframe> m_subframes;
};

#endif
