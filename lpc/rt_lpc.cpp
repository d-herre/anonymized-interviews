/*----------------------------------------------------------------------------
    rt_lpc - real-time LPC analysis/synthesis and visualization

    Copyright (c) 2004 Ananya Misra, Ge Wang, Perry R. Cook.  All rights reserved.
      http://soundlab.cs.princeton.edu/

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
    U.S.A.
-----------------------------------------------------------------------------*/
//-----------------------------------------------------------------------------
// name: rt_lpc.cpp
// desc: LPC voice disguise: shifts the formants and pitch of a voice by LPC
//       analysis and resynthesis. Headless; streams 16-bit PCM from stdin to
//       stdout, or processes a WAV file.
//
// author: Ananya Misra (amisra@cs.princeton.edu)
//         Ge Wang (gewang@cs.princeton.edu)
//         Perry Cook (prc@cs.princeton.edu)
//         (display, sound card and MIDI support removed; residual excitation,
//         formant shifting, streaming and control added)
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// How the program fits together
//
// main() reads the input one block at a time, runs process_block() on it,
// and writes one block of output, until the input ends:
//   input   16-bit little-endian mono PCM on stdin (the default), or a WAV
//           file (--file), resampled to the processing rate
//   output  16-bit PCM on stdout (the default), or a WAV file (--out)
// Streaming from stdin, it runs exactly as fast as the input arrives, and
// the output trails the input by the processing latency. Processing a WAV
// file, the output is lined up with the input instead: the latency is
// skipped at the start and made up with silence at the end, so the output
// has the same length and timing as the input.
//
// --control-fd starts one more thread, which reads setting commands and
// changes the voice settings under g_settings_mutex; process_block() takes
// a consistent copy of them at the start of every block.
//
// Two processing modes, chosen by do_ola (--ola<0|1>):
//
//   do_ola == false  each block is analyzed and resynthesized on its own.
//                    Simple, but the output jumps at block boundaries.
//
//   do_ola == true   overlap-add (the default). The block is half of the
//                    analysis window: each window is [previous block | new
//                    block] (50% overlap), and consecutive resynthesized
//                    windows are crossfaded using the two halves of
//                    g_window. The two filters being crossfaded share one
//                    continuous excitation and filter history, so the pitch
//                    pulses line up (see the resynthesis step in
//                    process_block()).
//
// Terminology: "pitch" here (and in lpc.cpp) is a pitch PERIOD in samples,
// not a frequency. A pitch of 0 means unvoiced (noise-like, e.g. "s", "f").
//-----------------------------------------------------------------------------
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <thread>
#include <mutex>
#include <system_error>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>

#include "lpc.h"
#include "psola.h"
#include "audio_io.h"
#include "silk_analysis.h"

// analysis window, in samples. With overlap-add the block (hop) is half
// of it; without, the block is the whole window.
#define FRAME_SIZE          1024
// most coefficients the synthesis keeps filter state for (the order is
// capped at 100 where it's set)
#define OLA_MAX_ORDER       128
// (residual excitation, OLA) the filter glides from one frame's to the next
// in steps this long
#define RESID_SUBFRAME      32

// pitch range the analysis looks for: down to just below the lowest adult
// male speaking voices (which reach down to roughly 60-70 Hz)
#define PITCH_LOWEST_HZ     55.0f
#define PITCH_HIGHEST_HZ    1200.0f




//-----------------------------------------------------------------------------
// global variables
//-----------------------------------------------------------------------------
int g_srate = 24000;            // processing sample rate (--srate)
bool do_ola = true;             // overlap-add (--ola)
int g_hop = 0;                  // block size: FRAME_SIZE / 2 with OLA, else FRAME_SIZE
float g_window[FRAME_SIZE];     // Hann window for the OLA crossfade
lpc_data g_lpc = NULL;          // analyzer/synthesizer state (lpc.cpp)
SilkAnalysis * g_silk = NULL;   // SILK's VAD and pitch estimator, fed every block

const char * g_wav_in = NULL;   // --file: input WAV (otherwise stdin)
const char * g_wav_out = NULL;  // --out: output WAV (otherwise stdout)
int g_control_fd = -1;          // --control-fd: read setting commands from here
int g_report_fd = -1;           // --report-fd: write per-frame voice measurements here

// ---- voice settings ----
// Changed by command-line options and the control thread, read by
// process_block(). Change them only while holding g_settings_mutex, and
// read them through get_voice_settings().
std::mutex g_settings_mutex;
float g_speed = 1.0f;           // "pitch factor": the detected pitch period is divided
                                //   by this, so > 1 raises the output pitch
float g_formant = 1.0f;         // "formant factor": every formant frequency is multiplied
                                //   by this. > 1 sounds like a shorter vocal tract,
                                //   < 1 like a longer one.
int g_formant_method = 0;       // how formants are shifted: 0 envelope warp, 1 poles,
                                //   2 line spectral frequencies
static const char * const g_formant_method_names[] = { "envelope", "poles", "lsf" };
int g_order = 30;               // LPC order: number of predictor coefficients
float g_out_gain_db = 0.0f;     // output gain in dB
// what drives the synthesis filter
enum { EXCITE_GLOTTAL = 0,      // recorded glottal pulse, once per period (+ noise
                                //   when unvoiced)
       EXCITE_IMPULSE,          // impulse train (+ noise when unvoiced)
       EXCITE_RESIDUAL,         // the voice's own LPC residual: the input,
                                //   pitch-shifted (psola.cpp) and whitened. Most
                                //   natural; one block more latency.
       EXCITE_COUNT };
int g_excitation = EXCITE_GLOTTAL;
static const char * const g_excitation_names[EXCITE_COUNT] = { "glottal", "impulse", "residual" };
bool g_balance = false;         // pre-/de-emphasis filters around the LPC
// how the voice is analyzed: LPC coefficients and pitch
enum { ANALYSIS_CLASSIC = 0,    // lpc_analyze(): autocorrelation fit (Hann window,
                                //   Levinson-Durbin) and autocorrelation pitch
       ANALYSIS_SILK,           // the SILK codec's (silk_analysis.cpp): Burg's
                                //   method, and its VAD and pitch estimator
       ANALYSIS_COUNT };
int g_analysis = ANALYSIS_CLASSIC;
static const char * const g_analysis_names[ANALYSIS_COUNT] = { "classic", "silk" };




//-----------------------------------------------------------------------------
// name: usage()
// desc: prints the command-line options
//-----------------------------------------------------------------------------
void usage()
{
    fprintf( stderr,
        "usage: rt_lpc [options]\n"
        "Reads 16-bit little-endian mono PCM on stdin and writes the processed PCM to\n"
        "stdout, until stdin ends; or processes a WAV file.\n"
        "  --srate=<N>            processing sample rate (default 24000)\n"
        "  --ola=<0|1>            overlap-add off/on (default on)\n"
        "  --file=<in.wav>        read this WAV file instead of stdin (resampled to the\n"
        "                         processing rate); the output is lined up with it\n"
        "  --out=<out.wav>        write a WAV file instead of PCM to stdout\n"
        "  --stdio                accepted for compatibility (it's the default)\n"
        "voice settings:\n"
        "  --formant=<f>          formant factor, 0.5 .. 2.0 (default 1)\n"
        "  --method=<m>           formant method: envelope, poles or lsf (default envelope)\n"
        "  --pitch=<f>            pitch factor, 0.05 .. 8 (default 1)\n"
        "  --order=<n>            LPC order, 1 .. 100 (default 30)\n"
        "  --excitation=<e>       what drives the filter: glottal (recorded glottal\n"
        "                         pulses, the default), impulse (impulse train) or\n"
        "                         residual (the voice's own LPC residual, pitch-\n"
        "                         shifted: more natural, one block more latency)\n"
        "  --impulse              same as --excitation=impulse\n"
        "  --emphasis             pre-/de-emphasis filters around the LPC\n"
        "  --analysis=<a>         how the voice is analyzed: classic (autocorrelation\n"
        "                         LPC and pitch, the default) or silk (the SILK codec's\n"
        "                         Burg LPC, voice activity detector and pitch estimator)\n"
        "  --gain=<dB>            output gain, -40 .. +40 dB (default 0)\n"
        "real-time control:\n"
        "  --control-fd=<n>       read setting commands from file descriptor n while\n"
        "                         running, one per line: \"<name> <value>\", where name is\n"
        "                         formant, method, pitch, order, gain (same values as the\n"
        "                         options above), excitation (glottal|impulse|\n"
        "                         residual), emphasis (on|off) or analysis (classic|\n"
        "                         silk). E.g. \"formant 1.2\"\n"
        "  --report-fd=<n>        write voice measurements to file descriptor n, one\n"
        "                         JSON line per frame: {\"f0\":Hz (0 = unvoiced),\n"
        "                         \"level\":dBFS, \"formants\":[[Hz, bandwidth Hz], ...]}\n"
        "Options take their value as --name=value, --namevalue or --name value.\n" );
}




//-----------------------------------------------------------------------------
// name: option_value()
// desc: if argv[*n] is the option `name`, returns its value, accepting
//       "--name=value", "--namevalue" and "--name value" (which consumes the
//       next argument). Otherwise returns NULL. Exits with usage() if the
//       option is given without a value.
//-----------------------------------------------------------------------------
const char * option_value( int argc, char ** argv, int * n, const char * name )
{
    size_t len = strlen( name );
    const char * arg = argv[*n];
    if( strncmp( arg, name, len ) != 0 ) return NULL;
    const char * v = arg + len;
    if( *v == '=' ) v++;
    else if( *v == '\0' && *n + 1 < argc ) v = argv[++(*n)];
    if( !*v )
    {
        fprintf( stderr, "rt_lpc: %s needs a value\n", name );
        usage();
        exit( 1 );
    }
    return v;
}




//-----------------------------------------------------------------------------
// name: VoiceSettings / get_voice_settings()
// desc: a consistent copy of all the voice settings, taken under
//       g_settings_mutex
//-----------------------------------------------------------------------------
struct VoiceSettings
{
    int order;
    float speed;          // pitch factor
    float formant;        // formant factor
    int method;           // formant method
    float gain_db;        // output gain
    int excitation;       // EXCITE_*
    bool balance;         // pre-/de-emphasis
    int analysis;         // ANALYSIS_*
};

VoiceSettings get_voice_settings( )
{
    std::lock_guard<std::mutex> lock( g_settings_mutex );
    VoiceSettings v;
    v.order = g_order;
    v.speed = g_speed;
    v.formant = g_formant;
    v.method = g_formant_method;
    v.gain_db = g_out_gain_db;
    v.excitation = g_excitation;
    v.balance = g_balance;
    v.analysis = g_analysis;
    return v;
}




//-----------------------------------------------------------------------------
// name: apply_setting()
// desc: changes one voice setting by name, from a text value, after
//       checking it. Used for both command-line options and control
//       commands, so both accept exactly the same names and values:
//         formant <0.5 .. 2.0>       method <envelope|poles|lsf>
//         pitch <0.05 .. 8>          order <1 .. 100>
//         gain <-40 .. 40> (dB)      excitation <glottal|impulse|residual>
//         emphasis <on|off>          analysis <classic|silk>
//       Returns false, with a message in err, if the name or value is not
//       valid; nothing is changed then.
//-----------------------------------------------------------------------------
static bool parse_number( const char * text, double lo, double hi, double * out )
{
    char * end;
    double v = strtod( text, &end );
    if( end == text || *end != '\0' || !( v >= lo && v <= hi ) ) return false;
    *out = v;
    return true;
}

bool apply_setting( const char * name, const char * value, char * err, size_t errlen )
{
    double v;
    std::lock_guard<std::mutex> lock( g_settings_mutex );

    if( strcmp( name, "formant" ) == 0 )
    {
        if( !parse_number( value, .5, 2.0, &v ) ) goto bad;
        g_formant = (float)v;
    }
    else if( strcmp( name, "method" ) == 0 )
    {
        if( strcmp( value, "envelope" ) == 0 ) g_formant_method = 0;
        else if( strcmp( value, "poles" ) == 0 ) g_formant_method = 1;
        else if( strcmp( value, "lsf" ) == 0 || strcmp( value, "LSF" ) == 0 ) g_formant_method = 2;
        else goto bad;
    }
    else if( strcmp( name, "pitch" ) == 0 )
    {
        if( !parse_number( value, .05, 8.0, &v ) ) goto bad;
        g_speed = (float)v;
    }
    else if( strcmp( name, "order" ) == 0 )
    {
        if( !parse_number( value, 1, 100, &v ) || v != (int)v ) goto bad;
        g_order = (int)v;
    }
    else if( strcmp( name, "gain" ) == 0 )
    {
        if( !parse_number( value, -40, 40, &v ) ) goto bad;
        g_out_gain_db = (float)v;
    }
    else if( strcmp( name, "excitation" ) == 0 )
    {
        int e = 0;
        while( e < EXCITE_COUNT && strcmp( value, g_excitation_names[e] ) != 0 ) e++;
        if( e == EXCITE_COUNT ) goto bad;
        g_excitation = e;
    }
    else if( strcmp( name, "emphasis" ) == 0 )
    {
        if( strcmp( value, "on" ) == 0 ) g_balance = true;
        else if( strcmp( value, "off" ) == 0 ) g_balance = false;
        else goto bad;
    }
    else if( strcmp( name, "analysis" ) == 0 )
    {
        int a = 0;
        while( a < ANALYSIS_COUNT && strcmp( value, g_analysis_names[a] ) != 0 ) a++;
        if( a == ANALYSIS_COUNT ) goto bad;
        g_analysis = a;
    }
    else
    {
        snprintf( err, errlen, "unknown setting '%s'", name );
        return false;
    }
    return true;

bad:
    snprintf( err, errlen, "invalid value '%s' for %s", value, name );
    return false;
}




//-----------------------------------------------------------------------------
// name: setting_from_option()
// desc: applies a voice setting given on the command line; exits with a
//       message if the value is invalid
//-----------------------------------------------------------------------------
void setting_from_option( const char * name, const char * value )
{
    char err[160];
    if( !apply_setting( name, value, err, sizeof(err) ) )
    {
        fprintf( stderr, "rt_lpc: %s\n", err );
        exit( 1 );
    }
}




//-----------------------------------------------------------------------------
// name: control_command()
// desc: one control line: "<name> <value>" (see apply_setting()). Blank
//       lines and lines starting with '#' are ignored; anything invalid is
//       reported on stderr and ignored.
//-----------------------------------------------------------------------------
static void control_command( const char * line )
{
    char name[32], value[64], extra[2], err[160];
    int fields = sscanf( line, "%31s %63s %1s", name, value, extra );
    if( fields <= 0 || name[0] == '#' ) return;
    if( fields != 2 )
    {
        fprintf( stderr, "rt_lpc: control: expected \"<name> <value>\"\n" );
        return;
    }
    if( !apply_setting( name, value, err, sizeof(err) ) )
        fprintf( stderr, "rt_lpc: control: %s\n", err );
}




//-----------------------------------------------------------------------------
// name: control_thread_func()
// desc: real-time control (--control-fd). Reads commands from fd, one per
//       line, "<name> <value>" (see apply_setting()), and applies each one;
//       process_block() picks the change up at the start of its next block.
//       Ends when fd reaches end of file.
//-----------------------------------------------------------------------------
void control_thread_func( int fd )
{
    // Plain read() rather than stdio: exit() locks every stdio FILE to flush
    // it, so a thread blocked in fgets() would stop the program from exiting.
    char buf[256];
    size_t len = 0;
    bool skipping = false;          // discarding the rest of an overlong line

    while( true )
    {
        ssize_t k = read( fd, buf + len, sizeof(buf) - 1 - len );
        if( k < 0 && errno == EINTR ) continue;
        if( k <= 0 ) break;                         // end of file (or error)
        len += (size_t)k;

        // handle every complete line in the buffer
        char * start = buf;
        char * nl;
        while( ( nl = (char *)memchr( start, '\n', buf + len - start ) ) )
        {
            *nl = '\0';
            if( !skipping ) control_command( start );
            skipping = false;
            start = nl + 1;
        }
        len -= (size_t)( start - buf );
        memmove( buf, start, len );

        // a full buffer with no newline: the line is too long
        if( len == sizeof(buf) - 1 )
        {
            if( !skipping ) fprintf( stderr, "rt_lpc: control: line too long, ignored\n" );
            skipping = true;
            len = 0;
        }
    }
    close( fd );
}




//=============================================================================
// Voice measurement report (--report-fd)
//
// For each analysis frame, writes one line of JSON to the report descriptor:
//     {"f0":118.4,"level":-23.1,"formants":[[640,82],[1110,95],[2480,140]]}
//   f0        fundamental frequency in Hz, 0 if the frame isn't voiced
//   level     RMS level of the frame in dBFS
//   formants  resonance candidates [frequency Hz, bandwidth Hz], lowest first
//             (only in voiced frames). Which is F1, F2, F3 is left to the
//             reader, which can use the whole utterance to decide.
//
// This is a separate measurement from the analysis rt_lpc resynthesizes
// from, which is tuned for that rather than for measuring. Here:
//   pitch     YIN (de Cheveigne & Kawahara 2002) over 60 .. 400 Hz
//   formants  lpc_measure_formants(): the usual formant-tracking recipe
//
// The descriptor is non-blocking: if the reader falls behind, lines are
// dropped rather than holding up the audio.
//=============================================================================
#define REPORT_F0_MIN       60.0
#define REPORT_F0_MAX       400.0
#define REPORT_YIN_THRESH   0.15      // YIN's aperiodicity threshold
#define REPORT_MIN_LEVEL    -60.0     // quieter frames count as unvoiced (dBFS)
#define REPORT_MAX_CANDS    6

// YIN pitch estimate of x[0..n-1], or 0 if no clear period
static double report_pitch( const SAMPLE * x, int n, double fs )
{
    const int tau_min = (int)( fs / REPORT_F0_MAX );
    const int tau_max = (int)( fs / REPORT_F0_MIN );
    const int w = n - tau_max;                    // integration window
    if( w < tau_max / 2 || tau_max >= 1024 ) return 0.0;

    static double d[1024];
    // difference function, then cumulative mean normalized
    d[0] = 1.0;
    double running = 0.0;
    for( int tau = 1; tau <= tau_max; tau++ )
    {
        double s = 0.0;
        for( int j = 0; j < w; j++ )
        {
            double diff = x[j] - x[j + tau];
            s += diff * diff;
        }
        running += s;
        d[tau] = running > 0.0 ? s * tau / running : 1.0;
    }

    // first dip below the threshold (then to the bottom of that dip)
    for( int tau = tau_min; tau <= tau_max; tau++ )
    {
        if( d[tau] < REPORT_YIN_THRESH )
        {
            while( tau + 1 <= tau_max && d[tau + 1] < d[tau] ) tau++;
            // parabolic interpolation around the minimum
            double t = tau;
            if( tau > tau_min && tau < tau_max )
            {
                double a = d[tau - 1], b = d[tau], c = d[tau + 1];
                double den = a - 2.0 * b + c;
                if( den > 0.0 ) t += 0.5 * ( a - c ) / den;
            }
            return fs / t;
        }
    }
    return 0.0;
}

// measures one frame and writes its line to g_report_fd
static void report_frame( const SAMPLE * x, int n, double fs )
{
    double ss = 0.0;
    for( int i = 0; i < n; i++ ) ss += (double)x[i] * x[i];
    double level = 10.0 * log10( ss / n + 1e-20 );

    double f0 = level > REPORT_MIN_LEVEL ? report_pitch( x, n, fs ) : 0.0;

    char line[512];
    int len = snprintf( line, sizeof(line), "{\"f0\":%.1f,\"level\":%.1f,\"formants\":[", f0, level );
    if( f0 > 0.0 )
    {
        double freq[REPORT_MAX_CANDS], bw[REPORT_MAX_CANDS];
        int k = lpc_measure_formants( x, n, fs, freq, bw, REPORT_MAX_CANDS );
        for( int i = 0; i < k; i++ )
            len += snprintf( line + len, sizeof(line) - len, "%s[%.0f,%.0f]", i ? "," : "", freq[i], bw[i] );
    }
    len += snprintf( line + len, sizeof(line) - len, "]}\n" );

    // all or nothing: a full pipe drops this line (lines are short, so a
    // partial write can only happen if the pipe is nearly full)
    ssize_t w = write( g_report_fd, line, len );
    if( w >= 0 && w < len )
    {
        // finish the line so the reader never sees a broken one
        const char * rest = line + w;
        size_t left = len - w;
        while( left > 0 )
        {
            ssize_t k = write( g_report_fd, rest, left );
            if( k > 0 ) { rest += k; left -= k; }
            else if( k < 0 && errno == EINTR ) continue;
            else if( k < 0 && errno == EAGAIN ) usleep( 1000 );
            else break;
        }
    }
}




//-----------------------------------------------------------------------------
// name: process_block()
// desc: processes one block: g_hop samples of input in, g_hop samples of
//       output out. In order:
//         1. build the analysis window
//         2. analysis: LPC coefficients, gain and pitch, either classic
//            (lpc_analyze()) or SILK's
//         3. formant shift (warp_formants*())
//         4. the residual excitation: pitch-shifted input, whitened
//         5. resynthesis, plus the overlap-add crossfade in OLA mode
//         6. output gain
//       The output trails the input by latency_samples().
//-----------------------------------------------------------------------------
void process_block( const SAMPLE * in, SAMPLE * out )
{
    const int hop = g_hop;                // samples per block
    const int N = FRAME_SIZE;             // analysis window: 2 * hop in OLA mode, else hop

    // working buffers (static because they're large)
    static SAMPLE window[FRAME_SIZE];     // analysis window
    static SAMPLE synth[FRAME_SIZE];      // resynthesized block
    static float coefs[OLA_MAX_ORDER];    // LPC coefficients
    // OLA carry-over from one block to the next (the synthesis keeps its own,
    // below)
    static SAMPLE prev_in[FRAME_SIZE];    // previous input block: first half of the next window

    float pitch, power;
    int i;

    // read the settings once per block. The control thread can change them
    // at any moment; using one consistent set per block matters most for
    // the order, which the analysis and the synthesis must agree on.
    const VoiceSettings vs = get_voice_settings();
    const int order = vs.order;
    const float speed = vs.speed;
    const float formant = vs.formant;
    const int method = vs.method;
    const int excitation = vs.excitation;
    const bool balance = vs.balance;
    const float gain = powf( 10.0f, vs.gain_db / 20.0f );

    // input samples before this block (the SILK analysis is addressed by
    // position in the input)
    static long block_start = 0;
    // SILK's analysis runs on every block, whatever the setting, so its
    // state (noise estimates, previous lag) is current when it's chosen
    g_silk->push( in, hop );

    // ---- analysis window ----
    // the raw input sample just before window[0] (for the residual)
    static SAMPLE raw_before_window = 0;
    const SAMPLE before_window = raw_before_window;
    if( do_ola )
    {
        // the next window starts after this window's first block
        raw_before_window = prev_in[hop-1];
        // [previous block | this block]: 50% overlap with the last window
        memcpy( window, prev_in, hop * sizeof(SAMPLE) );
        memcpy( window + hop, in, hop * sizeof(SAMPLE) );
        // this block becomes the first half of the next window
        memcpy( prev_in, in, hop * sizeof(SAMPLE) );
    }
    else
    {
        memcpy( window, in, N * sizeof(SAMPLE) );
        raw_before_window = in[N-1];
    }

    // ---- measurement report (--report-fd), on the frame as it came in ----
    if( g_report_fd >= 0 )
        report_frame( window, N, g_srate );

    // ---- analysis ----
    if( balance )
        lpc_preemphasis( window, N, .5f );   // optional pre-emphasis
    if( vs.analysis == ANALYSIS_SILK )
    {
        // LPC coefficients by Burg's method (no window needed), the gain
        // the excitation expects (as lpc_analyze() measures it), and the
        // pitch from SILK's estimator over the block this frame's pitch is
        // used for: the older block of an OLA window (the one the output
        // block and the residual below are for), or the block itself
        SilkAnalysis::lpc( window, N, coefs, order );
        power = lpc_predict( g_lpc, window, N, coefs, order, NULL );
        const long a = do_ola ? block_start - hop : block_start;
        pitch = g_silk->pitch_over( a, a + hop );
    }
    else
        // coefficients, gain (power), pitch period
        lpc_analyze( g_lpc, window, N, coefs, order, &power, &pitch, NULL );
    // what the analysis found, before the formant shift changes it (the
    // residual excitation needs the original filter)
    const float analysis_power = power;
    static float analysis_coefs[OLA_MAX_ORDER];
    memcpy( analysis_coefs, coefs, order * sizeof(float) );

    // ---- formant shift ----
    // move every formant by the formant factor, keeping the loudness the
    // same, with the chosen method. At exactly 1.0 the coefficients are
    // left untouched.
    if( formant != 1.0f )
    {
        if( method == 1 )
            // bw_factor 1: keep bandwidths the same in Hz
            warp_formants_poles( coefs, order, (float)g_srate, formant, &power, NULL, NULL, 1.0f );
        else if( method == 2 )
            warp_formants_lsf( coefs, order, (float)g_srate, formant, &power );
        else
            warp_formants( coefs, order, (float)g_srate, formant, &power );
    }

    // ---- residual excitation (pitch-shifted) ----
    // Instead of an artificial pulse train, the voice's own excitation
    // can drive the formant-shifted filter:
    //   1. PsolaShifter (psola.cpp) shifts the PITCH of the input
    //      itself, a period at a time, keeping its formants. It places
    //      its pitch marks on the peaks of the LPC residual (the "guide").
    //   2. The shifted voice is whitened with the ORIGINAL analysis
    //      filter (the residual: the voice with its vocal tract taken
    //      out), scaled by however much the formant shift changed the
    //      gain, ...
    //   3. ... and recolored with the formant-shifted filter (in the
    //      resynthesis step below).
    // Steps 2 and 3 use the filters for the moment each sample is
    // PLAYED, so no matter how far a period was moved, it's whitened
    // and recolored by the same frame's filters. With no formant or
    // pitch change the whole thing undoes itself exactly, so the voice
    // comes through unchanged.
    //
    // OLA: a frame's window is centered on the boundary between its two
    // blocks, so its filter is the right one THERE. The filters glide
    // from one frame's to the next, a step every RESID_SUBFRAME samples,
    // by interpolating reflection coefficients (always stable): the
    // older block in this window (whose residual guides the pitch
    // marks) glides from the previous frame to this one, and the block
    // PsolaShifter hands back, one block later, from two frames ago to
    // the previous one.
    // No overlap: the frame is the block itself, so each block uses its
    // own frame's filters.
    //
    // This runs on every block, whatever the excitation setting, so the
    // streams have no gaps when it's switched on.
    static SAMPLE xhist[OLA_MAX_ORDER + FRAME_SIZE];  // [history | block]: input
    static SAMPLE shist[OLA_MAX_ORDER + FRAME_SIZE];  // [history | block]: shifted
    static SAMPLE guide[FRAME_SIZE];                  // residual of the input block
    static SAMPLE resid_exc[FRAME_SIZE];              // residual of the shifted block
    double shifted_energy;                            // energy the output should have
    static PsolaShifter psola( hop, (int)ceilf( g_srate / PITCH_LOWEST_HZ ) + 1 );
    // the last three frames' filters, as reflection coefficients
    // (zero-padded to OLA_MAX_ORDER): analysis and formant-shifted, with
    // their order and gain change. [0] = two frames ago, [1] = the
    // previous frame, [2] = this one.
    static float k_an[3][OLA_MAX_ORDER], k_syn[3][OLA_MAX_ORDER], g_ch[3];
    static int ord_f[3];
    for( int f = 0; f < 2; f++ )
    {
        memcpy( k_an[f], k_an[f+1], sizeof k_an[f] );
        memcpy( k_syn[f], k_syn[f+1], sizeof k_syn[f] );
        g_ch[f] = g_ch[f+1];
        ord_f[f] = ord_f[f+1];
    }
    memset( k_an[2], 0, sizeof k_an[2] );
    memset( k_syn[2], 0, sizeof k_syn[2] );
    lpc_to_reflection( analysis_coefs, order, k_an[2] );
    lpc_to_reflection( coefs, order, k_syn[2] );
    ord_f[2] = order;
    g_ch[2] = analysis_power > 0 ? power / analysis_power : 0.0f;
    // which frames' filters glide across the input block (guide) and
    // across the shifted block (output): from frame A to frame B
    const int guide_a = do_ola ? 1 : 2, guide_b = 2;
    const int out_a = do_ola ? 0 : 1, out_b = 1;
    // the filter a fraction t of the way from frame fa's to frame fb's
    auto filter_at = []( const float (*k)[OLA_MAX_ORDER], int fa, int fb,
                         float t, int ord, float * a )
    {
        float kk[OLA_MAX_ORDER];
        for( int j = 0; j < ord; j++ )
            kk[j] = k[fa][j] + t * ( k[fb][j] - k[fa][j] );
        reflection_to_lpc( kk, ord, a );
    };
    {
        // the input block: window[0 .. hop), which is the older block in
        // an OLA window, or the whole frame otherwise. As analyzed:
        // pre-emphasized if that's on, in which case the first sample
        // needs the sample before it, which lpc_preemphasis() didn't have.
        SAMPLE * x = xhist + OLA_MAX_ORDER;
        memcpy( x, window, hop * sizeof(SAMPLE) );
        if( balance ) x[0] -= .5f * before_window;

        // its residual, to guide the pitch marks
        float a[OLA_MAX_ORDER];
        int ord = std::max( ord_f[guide_a], ord_f[guide_b] );
        for( int s0 = 0; s0 < hop; s0 += RESID_SUBFRAME )
        {
            const int len = std::min( RESID_SUBFRAME, hop - s0 );
            filter_at( k_an, guide_a, guide_b, do_ola ? ( s0 + .5f * len ) / hop : 0.0f, ord, a );
            for( i = s0; i < s0 + len; i++ )
            {
                float e = x[i];
                for( int j = 0; j < ord; j++ ) e -= a[j] * x[i - 1 - j];
                guide[i] = e;
            }
        }

        // pitch-shift it: the period measured (0 if unvoiced) and the
        // one wanted, divided by the pitch factor. A block comes back
        // from one block earlier.
        psola.push( x, guide, hop, pitch, pitch / speed );
        SAMPLE * y = shist + OLA_MAX_ORDER;
        psola.pull( y, hop );

        // the shifted block's residual, by the analysis filters for its
        // own time, scaled by the gain change (the formant shift's
        // loudness correction). Also the shifted block's energy: what the
        // output should have (see below).
        shifted_energy = 0.0;
        ord = std::max( ord_f[out_a], ord_f[out_b] );
        for( int s0 = 0; s0 < hop; s0 += RESID_SUBFRAME )
        {
            const int len = std::min( RESID_SUBFRAME, hop - s0 );
            filter_at( k_an, out_a, out_b, do_ola ? ( s0 + .5f * len ) / hop : 0.0f, ord, a );
            for( i = s0; i < s0 + len; i++ )
            {
                float e = y[i];
                for( int j = 0; j < ord; j++ ) e -= a[j] * y[i - 1 - j];
                const float g = do_ola ? g_ch[out_a] + ( g_ch[out_b] - g_ch[out_a] ) * ( i + .5f ) / hop
                                       : g_ch[out_b];
                resid_exc[i] = e * g;
                shifted_energy += (double)y[i] * y[i];
            }
        }

        // keep the last OLA_MAX_ORDER samples of both as history
        memmove( xhist, xhist + hop, OLA_MAX_ORDER * sizeof(SAMPLE) );
        memmove( shist, shist + hop, OLA_MAX_ORDER * sizeof(SAMPLE) );
    }

    // when the excitation setting changes, the new path's filter
    // history is stale: start it from rest
    static int last_excitation = -1;
    const int prev_excitation = last_excitation;
    const bool excitation_changed = excitation != last_excitation;
    last_excitation = excitation;

    // ---- resynthesis ----
    // Dividing the period by the pitch factor raises the pitch when
    // speed > 1. The excitation is the recorded glottal pulse, an
    // impulse train, or the residual from above.
    if( excitation == EXCITE_RESIDUAL )
    {
        // residual: recolor the shifted block's residual (from above)
        // with the formant-shifted filters for its time, carrying the
        // filter's history over from block to block
        static float sstate[OLA_MAX_ORDER];
        static int sstate_order = 0;
        if( excitation_changed )
        {
            memset( sstate, 0, sizeof sstate );
            sstate_order = 0;
        }
        const int ord = std::max( ord_f[out_a], ord_f[out_b] );
        if( ord > sstate_order )
            memset( sstate + sstate_order, 0, ( ord - sstate_order ) * sizeof(float) );
        sstate_order = ord;
        float a[OLA_MAX_ORDER];
        for( int s0 = 0; s0 < hop; s0 += RESID_SUBFRAME )
        {
            const int len = std::min( RESID_SUBFRAME, hop - s0 );
            filter_at( k_syn, out_a, out_b, do_ola ? ( s0 + .5f * len ) / hop : 0.0f, ord, a );
            lpc_filter( resid_exc + s0, synth + s0, len, a, ord, sstate );
        }
        if( !std::isfinite( sstate[0] ) || fabsf( sstate[0] ) > 1e4f )
        {
            memset( sstate, 0, sizeof sstate );
            memset( synth, 0, hop * sizeof(SAMPLE) );
        }

        // loudness: the formant-shifted filter is meant to keep it, but
        // it can't always: a filter that's sharp enough to pick out
        // single harmonics (a high LPC order on a high voice) can make
        // the shifted harmonics land on its peaks, or between them, and
        // swing the level by 10 dB or more; and some formant methods'
        // own loudness correction is only rough. So scale each block to
        // the energy of the shifted voice it came from, within +/-12 dB,
        // ramping from the last block's correction. With no formant
        // change the filters cancel, the energies match, and the
        // correction is 1.
        static float prev_fix = 1.0f;
        if( excitation_changed ) prev_fix = 1.0f;
        {
            double out_energy = 0.0;
            for( i = 0; i < hop; i++ ) out_energy += (double)synth[i] * synth[i];
            const double floor = hop * 1e-8;          // -80 dBFS: leave silence alone
            float fix = (float)sqrt( ( shifted_energy + floor ) / ( out_energy + floor ) );
            fix = std::min( 4.0f, std::max( .25f, fix ) );
            for( i = 0; i < hop; i++ )
                synth[i] *= prev_fix + ( fix - prev_fix ) * ( i + 1 ) / hop;
            // the filter's history carries on from what it produced, not
            // the corrected block, so it isn't scaled
            prev_fix = fix;
        }

        // undo the pre-emphasis tilt. lpc_deemphasis() averages each
        // sample with the one before it; for sample 0 that's the last
        // block's final sample, kept here.
        static SAMPLE last_sample = 0;
        const SAMPLE s0 = synth[0], s_end = synth[hop-1];
        if( balance )
        {
            lpc_deemphasis( synth, hop, .5f );
            synth[0] = .5f * s0 + .5f * last_sample;
        }
        last_sample = s_end;
    }
    else if( do_ola )
    {
        // OLA: the output block is a crossfade between two filters,
        // the previous frame's (fading out) and this frame's (fading in).
        // Both must be driven by THE SAME excitation from THE SAME
        // filter history, or their pitch pulses land at different times
        // and the crossfade smears them into something less periodic
        // (and rougher) than either.
        //
        // So: generate one hop of excitation, continuing the pulse train
        // from the last block, and run it through
        //   - the previous frame's filter -> the previous window's second
        //     half (fades out), and
        //   - this frame's filter         -> this window's first half
        //     (fades in),
        // both starting from filt_mid: the previous filter's state where
        // its first half ended, i.e. exactly where this excitation
        // starts. The new filter's state afterwards becomes the next
        // filt_mid. The excitation and the filter history therefore run
        // on without a break; only the coefficients crossfade.
        static SAMPLE exc[FRAME_SIZE];        // this block's excitation
        static SAMPLE tail[FRAME_SIZE];       // previous filter's output
        static float prev_coefs[OLA_MAX_ORDER];
        static float filt_mid[OLA_MAX_ORDER]; // filter state at the window midpoint
        static float state[OLA_MAX_ORDER];
        static int prev_order = 0;
        if( prev_excitation == EXCITE_RESIDUAL )
            prev_order = 0;      // coming from the residual path: restart

        lpc_excitation( g_lpc, exc, hop, power, pitch / speed, excitation == EXCITE_GLOTTAL );
        // the filter's most recent output (its state is newest first)
        const SAMPLE prev_out = prev_order > 0 ? filt_mid[0] : 0.0f;

        // previous window's second half (silence on the very first block)
        if( prev_order > 0 )
        {
            memcpy( state, filt_mid, prev_order * sizeof(float) );
            lpc_filter( exc, tail, hop, prev_coefs, prev_order, state );
        }
        else
            memset( tail, 0, hop * sizeof(SAMPLE) );

        // this window's first half. If the order went up, the extra
        // (older) history the new filter wants doesn't exist: use zeros.
        if( order > prev_order )
            memset( filt_mid + prev_order, 0, ( order - prev_order ) * sizeof(float) );
        lpc_filter( exc, synth, hop, coefs, order, filt_mid );

        // a filter that blew up (a degenerate frame) would keep ringing
        // through the carried-over state. Drop it: silence this half,
        // and start the next block from rest with no previous filter.
        if( !std::isfinite( filt_mid[0] ) || fabsf( filt_mid[0] ) > 1e4f )
        {
            memset( filt_mid, 0, OLA_MAX_ORDER * sizeof(float) );
            memset( synth, 0, hop * sizeof(SAMPLE) );
            prev_order = 0;
        }
        else
        {
            memcpy( prev_coefs, coefs, order * sizeof(float) );
            prev_order = order;
        }

        if( balance )
        {
            // undo the pre-emphasis tilt. lpc_deemphasis() averages each
            // sample with the one before it; for sample 0 of both halves
            // that's the filter's last output, saved as prev_out above.
            const SAMPLE t0 = tail[0], s0 = synth[0];
            lpc_deemphasis( tail, hop, .5f );
            lpc_deemphasis( synth, hop, .5f );
            tail[0] = .5f * t0 + .5f * prev_out;
            synth[0] = .5f * s0 + .5f * prev_out;
        }

        // crossfade: the previous filter fades out with the falling half
        // of g_window while this one fades in with the rising half. The
        // two halves of a Hann window sum to 1, and because both outputs
        // share the excitation they add coherently: the result,
        // synth[0 .. hop), is exactly one block.
        for( i = 0; i < hop; i++ )
            synth[i] = synth[i] * g_window[i] + tail[i] * g_window[i + hop];
    }
    else
    {
        // no overlap: one window per block; lpc_synthesize() carries the
        // pulse timing and filter history across blocks itself
        lpc_synthesize( g_lpc, synth, N, coefs, order, power, pitch / speed,
                        excitation == EXCITE_GLOTTAL );
        if( balance )
            lpc_deemphasis( synth, N, .5f );     // undo the pre-emphasis tilt
    }

    // ---- output gain ----
    // ramped from the previous block's gain across this block, so that a
    // change doesn't click. It's applied to the finished block only,
    // never to anything carried over to the next one.
    static float prev_gain = 1.0f;
    for( i = 0; i < hop; i++ )
        out[i] = synth[i] * ( prev_gain + ( gain - prev_gain ) * ( i + 1 ) / hop );
    prev_gain = gain;

    block_start += hop;
}




//-----------------------------------------------------------------------------
// name: latency_samples()
// desc: how far process_block()'s output trails its input, in samples, with
//       the given excitation: with overlap-add a block's output is for the
//       block before it, and the residual excitation adds one more block
//       (PsolaShifter's lookahead)
//-----------------------------------------------------------------------------
long latency_samples( int excitation )
{
    return (long)g_hop * ( ( do_ola ? 1 : 0 ) + ( excitation == EXCITE_RESIDUAL ? 1 : 0 ) );
}




//-----------------------------------------------------------------------------
// Name: main( )
// Desc: parses the options, opens the input and output, and processes the
//       input block by block until it ends (or the output closes)
//-----------------------------------------------------------------------------
int main( int argc, char ** argv )
{
    bool stdio_flag = false;

    // parse command line arguments
    for( int n = 1; n < argc; n++ )
    {
        if( const char * v = option_value( argc, argv, &n, "--srate" ) )
        {
            g_srate = atoi( v );
            if( g_srate < 8000 || g_srate > 192000 )
            {
                fprintf( stderr, "rt_lpc: invalid sample rate '%s'\n", v );
                exit( 1 );
            }
        }
        else if( const char * v = option_value( argc, argv, &n, "--ola" ) )
            do_ola = atoi( v ) != 0;
        else if( const char * v = option_value( argc, argv, &n, "--file" ) )
            g_wav_in = v;
        else if( const char * v = option_value( argc, argv, &n, "--out" ) )
            g_wav_out = v;
        else if( strcmp( argv[n], "--stdio" ) == 0 )
            stdio_flag = true;
        else if( const char * v = option_value( argc, argv, &n, "--formant" ) )
            setting_from_option( "formant", v );
        else if( const char * v = option_value( argc, argv, &n, "--method" ) )
            setting_from_option( "method", v );
        else if( const char * v = option_value( argc, argv, &n, "--pitch" ) )
            setting_from_option( "pitch", v );
        else if( const char * v = option_value( argc, argv, &n, "--order" ) )
            setting_from_option( "order", v );
        else if( const char * v = option_value( argc, argv, &n, "--gain" ) )
            setting_from_option( "gain", v );
        else if( const char * v = option_value( argc, argv, &n, "--excitation" ) )
            setting_from_option( "excitation", v );
        else if( strcmp( argv[n], "--impulse" ) == 0 )
            setting_from_option( "excitation", "impulse" );
        else if( strcmp( argv[n], "--emphasis" ) == 0 )
            setting_from_option( "emphasis", "on" );
        else if( const char * v = option_value( argc, argv, &n, "--analysis" ) )
            setting_from_option( "analysis", v );
        else if( const char * v = option_value( argc, argv, &n, "--report-fd" ) )
        {
            g_report_fd = atoi( v );
            int flags = g_report_fd >= 0 ? fcntl( g_report_fd, F_GETFL ) : -1;
            if( flags == -1 )
            {
                fprintf( stderr, "rt_lpc: --report-fd %s is not an open file descriptor\n", v );
                exit( 1 );
            }
            // never let a slow reader hold up the audio
            fcntl( g_report_fd, F_SETFL, flags | O_NONBLOCK );
        }
        else if( const char * v = option_value( argc, argv, &n, "--control-fd" ) )
        {
            g_control_fd = atoi( v );
            if( g_control_fd < 0 || fcntl( g_control_fd, F_GETFD ) == -1 )
            {
                fprintf( stderr, "rt_lpc: --control-fd %s is not an open file descriptor\n", v );
                exit( 1 );
            }
        }
        else if( strcmp( argv[n], "--help" ) == 0 || strcmp( argv[n], "-h" ) == 0 )
        {
            usage();
            return 0;
        }
        else
        {
            fprintf( stderr, "rt_lpc: unknown option '%s'\n", argv[n] );
            usage();
            exit( 1 );
        }
    }
    if( stdio_flag && g_wav_in )
    {
        fprintf( stderr, "rt_lpc: --stdio and --file can't be used together\n" );
        exit( 1 );
    }

    // a closed output pipe should show up as a write error, not kill us
    signal( SIGPIPE, SIG_IGN );

    // without --out, stdout carries the audio. Keep a private copy of it for
    // that, and point file descriptor 1 at stderr, so that any message
    // printed to stdout can't end up in the audio stream.
    int audio_out_fd = -1;
    if( !g_wav_out )
    {
        fflush( stdout );
        audio_out_fd = dup( 1 );
        dup2( 2, 1 );
    }

    // the analysis and the crossfade window (periodic Hann: its two halves
    // add up to exactly 1)
    g_hop = do_ola ? FRAME_SIZE / 2 : FRAME_SIZE;
    for( int i = 0; i < FRAME_SIZE; i++ )
        g_window[i] = (float)( .5 - .5 * cos( 2.0 * M_PI * i / FRAME_SIZE ) );
    g_lpc = lpc_create( );
    lpc_set_pitch_range( g_lpc, (float)g_srate, PITCH_LOWEST_HZ, PITCH_HIGHEST_HZ );
    g_silk = new SilkAnalysis( g_srate );

    // input and output
    std::string err;
    AudioSource * source = g_wav_in ? open_wav_source( g_wav_in, g_srate, &err )
                                    : open_pcm_source( 0 );
    AudioSink * sink = g_wav_out ? open_wav_sink( g_wav_out, g_srate, &err )
                                 : open_pcm_sink( audio_out_fd );
    if( !source || !sink )
    {
        fprintf( stderr, "rt_lpc: %s\n", err.c_str() );
        return 1;
    }

    const VoiceSettings vs = get_voice_settings();
    fprintf( stderr, "rt_lpc: %d Hz, %s, %d-sample blocks; order %d, formant factor %.2f (%s), "
             "pitch factor %.2f, excitation %s, analysis %s, gain %+.1f dB; %s -> %s\n",
             g_srate, do_ola ? "overlap-add" : "no overlap", g_hop, vs.order, vs.formant,
             g_formant_method_names[vs.method], vs.speed, g_excitation_names[vs.excitation],
             g_analysis_names[vs.analysis], vs.gain_db, g_wav_in ? g_wav_in : "stdin", g_wav_out ? g_wav_out : "stdout" );

    // real-time control. Detached: it ends by itself when the other end of
    // the descriptor is closed, and must not hold up exit while it waits
    // for a command.
    if( g_control_fd >= 0 )
    {
        try { std::thread( control_thread_func, g_control_fd ).detach(); }
        catch( std::system_error & e )
        {
            fprintf( stderr, "rt_lpc: cannot start control thread (%s)\n", e.what() );
            return 1;
        }
    }

    // ---- the processing loop ----
    // A file is processed with its output lined up with it: the first
    // `skip` samples of output (the latency) are dropped, and the input is
    // followed by silence until as many samples have come out as went in.
    // A stream just goes block for block until it ends; a final partial
    // block is padded with silence.
    const bool aligned = source->length() >= 0;
    long skip = aligned ? latency_samples( vs.excitation ) : 0;
    long left = aligned ? source->length() : -1;    // output still to write
    long frames = 0;
    bool input_done = false;
    std::string end = "end of input";
    SAMPLE in[FRAME_SIZE], out[FRAME_SIZE];
    while( true )
    {
        int got = 0;
        if( input_done ) memset( in, 0, g_hop * sizeof(SAMPLE) );
        else got = source->read( in, g_hop );   // (zero-padded at the end)
        frames += got;
        if( got < g_hop ) input_done = true;
        if( got == 0 && !aligned ) break;

        process_block( in, out );

        int start = (int)std::min( skip, (long)g_hop );
        skip -= start;
        int n = g_hop - start;
        if( aligned ) n = (int)std::min( (long)n, left );
        if( n > 0 && !sink->write( out + start, n ) )
        {
            end = "output closed";
            break;
        }
        if( aligned && ( left -= n ) <= 0 ) break;
        if( !aligned && input_done ) break;
    }

    delete sink;                    // finishes a WAV file
    delete source;
    lpc_destroy( g_lpc );
    delete g_silk;
    fprintf( stderr, "rt_lpc: %s; done (%ld frames)\n", end.c_str(), frames );
    return 0;
}
