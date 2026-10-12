//-----------------------------------------------------------------------------
// name: audio_io.cpp
// desc: rt_lpc's audio input and output (see audio_io.h)
//-----------------------------------------------------------------------------
#include "audio_io.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <poll.h>
#include <unistd.h>

static inline unsigned int rd16( const unsigned char * p ) { return p[0] | ( p[1] << 8 ); }
static inline unsigned int rd32( const unsigned char * p )
{ return p[0] | ( p[1] << 8 ) | ( p[2] << 16 ) | ( (unsigned int)p[3] << 24 ); }
static inline void wr16( unsigned char * p, unsigned int v ) { p[0] = v & 0xff; p[1] = ( v >> 8 ) & 0xff; }
static inline void wr32( unsigned char * p, unsigned int v )
{ p[0] = v & 0xff; p[1] = ( v >> 8 ) & 0xff; p[2] = ( v >> 16 ) & 0xff; p[3] = ( v >> 24 ) & 0xff; }

// float sample -> 16-bit, clipped
static inline short to_s16( float v )
{
    v = v > 1.0f ? 1.0f : ( v < -1.0f ? -1.0f : v );
    return (short)lrintf( v * 32767.0f );
}

// waits until fd is ready for `events` (for descriptors in non-blocking mode)
static void wait_for( int fd, short events )
{
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = events;
    pfd.revents = 0;
    poll( &pfd, 1, 100 );
}




//=============================================================================
// raw PCM on a file descriptor
//=============================================================================
class PcmSource : public AudioSource
{
public:
    PcmSource( int fd ) : m_fd( fd ), m_done( false ) {}

    int read( float * out, int n )
    {
        if( m_done ) return 0;
        m_buf.resize( (size_t)n * 2 );
        size_t want = (size_t)n * 2, got = 0;
        while( got < want )
        {
            ssize_t k = ::read( m_fd, &m_buf[got], want - got );
            if( k > 0 ) { got += (size_t)k; continue; }
            if( k < 0 && errno == EINTR ) continue;
            if( k < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) ) { wait_for( m_fd, POLLIN ); continue; }
            m_done = true;                  // end of file (writer closed), or an error
            break;
        }
        int frames = (int)( got / 2 );      // (a dangling odd byte is dropped)
        for( int i = 0; i < frames; i++ )
            out[i] = (short)rd16( &m_buf[(size_t)i * 2] ) / 32768.0f;
        for( int i = frames; i < n; i++ )
            out[i] = 0.0f;
        return frames;
    }

private:
    int m_fd;
    bool m_done;
    std::vector<unsigned char> m_buf;
};


class PcmSink : public AudioSink
{
public:
    PcmSink( int fd ) : m_fd( fd ) {}

    bool write( const float * in, int n )
    {
        m_buf.resize( (size_t)n * 2 );
        for( int i = 0; i < n; i++ )
            wr16( &m_buf[(size_t)i * 2], (unsigned short)to_s16( in[i] ) );
        size_t done = 0;
        while( done < m_buf.size() )
        {
            ssize_t k = ::write( m_fd, &m_buf[done], m_buf.size() - done );
            if( k > 0 ) { done += (size_t)k; continue; }
            if( k < 0 && errno == EINTR ) continue;
            if( k < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) ) { wait_for( m_fd, POLLOUT ); continue; }
            return false;                   // EPIPE etc.: the reader is gone
        }
        return true;
    }

private:
    int m_fd;
    std::vector<unsigned char> m_buf;
};


AudioSource * open_pcm_source( int fd ) { return new PcmSource( fd ); }
AudioSink * open_pcm_sink( int fd ) { return new PcmSink( fd ); }




//=============================================================================
// WAV files
//=============================================================================

//-----------------------------------------------------------------------------
// name: resample()
// desc: x at rate `from` -> rate `to`, by band-limited interpolation: each
//       output sample is the input around its position weighted by a
//       Hann-windowed sinc (16 zero crossings each side), with the cutoff
//       at 95% of the lower of the two Nyquist frequencies, so downsampling
//       doesn't alias
//-----------------------------------------------------------------------------
static std::vector<float> resample( const std::vector<float> & x, int from, int to )
{
    if( from == to || x.empty() ) return x;
    const double ratio = (double)to / from;                  // output samples per input sample
    const double fc = 0.95 * std::min( 1.0, ratio );         // cutoff, relative to the input Nyquist
    const double half = 16.0 / fc;                           // kernel half-width, in input samples
    const long n_in = (long)x.size();
    const long n_out = (long)floor( n_in * ratio );
    std::vector<float> y( (size_t)n_out );
    for( long m = 0; m < n_out; m++ )
    {
        const double p = m / ratio;                          // position in the input
        long i0 = std::max( 0L, (long)ceil( p - half ) );
        long i1 = std::min( n_in - 1, (long)floor( p + half ) );
        double sum = 0.0;
        for( long i = i0; i <= i1; i++ )
        {
            const double d = p - i;
            const double t = fc * d;
            const double sinc = fabs( t ) < 1e-9 ? 1.0 : sin( M_PI * t ) / ( M_PI * t );
            const double w = 0.5 + 0.5 * cos( M_PI * d / half );
            sum += x[(size_t)i] * fc * sinc * w;
        }
        y[(size_t)m] = (float)sum;
    }
    return y;
}




//-----------------------------------------------------------------------------
// name: load_wav()
// desc: reads a whole WAV file and mixes it down to mono floats. Returns
//       false with a message in *err if it can't.
//-----------------------------------------------------------------------------
static bool load_wav( const char * path, std::vector<float> * out, int * out_rate, std::string * err )
{
    char msg[512];
    FILE * f = fopen( path, "rb" );
    if( !f )
    {
        snprintf( msg, sizeof(msg), "cannot open '%s': %s", path, strerror( errno ) );
        *err = msg;
        return false;
    }
    fseek( f, 0, SEEK_END );
    long file_size = ftell( f );
    fseek( f, 0, SEEK_SET );

    unsigned char hdr[12];
    if( fread( hdr, 1, 12, f ) != 12 || memcmp( hdr, "RIFF", 4 ) || memcmp( hdr + 8, "WAVE", 4 ) )
    {
        fclose( f );
        snprintf( msg, sizeof(msg), "'%s' is not a RIFF/WAVE file", path );
        *err = msg;
        return false;
    }

    // walk the chunks for "fmt " and "data"
    bool have_fmt = false, have_data = false;
    unsigned int tag = 0, channels = 0, rate = 0, block_align = 0, bits = 0;
    long data_offset = 0, data_size = 0;
    unsigned char ck[8];
    while( fread( ck, 1, 8, f ) == 8 )
    {
        unsigned int size = rd32( ck + 4 );
        long body = ftell( f );
        if( !memcmp( ck, "fmt ", 4 ) )
        {
            unsigned char fmt[40] = { 0 };
            size_t n = size < sizeof(fmt) ? size : sizeof(fmt);
            if( n < 16 || fread( fmt, 1, n, f ) != n ) break;
            tag         = rd16( fmt );
            channels    = rd16( fmt + 2 );
            rate        = rd32( fmt + 4 );
            block_align = rd16( fmt + 12 );
            bits        = rd16( fmt + 14 );
            // WAVE_FORMAT_EXTENSIBLE: the real tag starts the SubFormat GUID
            if( tag == 0xFFFE && n >= 26 ) tag = rd16( fmt + 24 );
            have_fmt = true;
        }
        else if( !memcmp( ck, "data", 4 ) )
        {
            data_offset = body;
            long remaining = file_size - body;
            // a size of 0 (or too large) comes from recorders that never
            // finished the header: use the rest of the file
            data_size = ( size == 0 || (long)size > remaining ) ? remaining : (long)size;
            have_data = true;
            if( have_fmt ) break;
        }
        long next = body + (long)size + ( size & 1 );      // chunks are word aligned
        if( next >= file_size || fseek( f, next, SEEK_SET ) != 0 ) break;
    }
    if( !have_fmt || !have_data )
    {
        fclose( f );
        snprintf( msg, sizeof(msg), "'%s' is missing its %s chunk", path, have_fmt ? "data" : "fmt " );
        *err = msg;
        return false;
    }

    const unsigned int bytes = channels ? block_align / channels : 0;
    const bool is_pcm = tag == 1 && bytes >= 1 && bytes <= 4;
    const bool is_float = tag == 3 && ( bytes == 4 || bytes == 8 );
    if( channels == 0 || rate == 0 || block_align == 0 || ( !is_pcm && !is_float ) )
    {
        fclose( f );
        snprintf( msg, sizeof(msg), "'%s': unsupported WAV encoding (format tag %u, %u bits, "
                  "%u channels); supported: PCM 8/16/24/32-bit, float 32/64-bit", path, tag, bits, channels );
        *err = msg;
        return false;
    }

    long frames = data_size / block_align;
    std::vector<unsigned char> raw( (size_t)std::max( frames, 0L ) * block_align );
    fseek( f, data_offset, SEEK_SET );
    size_t got = raw.empty() ? 0 : fread( &raw[0], 1, raw.size(), f );
    fclose( f );
    frames = (long)( got / block_align );
    if( frames <= 0 )
    {
        snprintf( msg, sizeof(msg), "'%s' contains no audio", path );
        *err = msg;
        return false;
    }

    out->assign( (size_t)frames, 0.0f );
    for( long i = 0; i < frames; i++ )
    {
        double sum = 0.0;
        for( unsigned int c = 0; c < channels; c++ )
        {
            const unsigned char * p = &raw[(size_t)i * block_align + c * bytes];
            double v;
            if( is_float )
            {
                if( bytes == 4 ) { unsigned int u = rd32( p ); float x; memcpy( &x, &u, 4 ); v = x; }
                else
                {
                    unsigned long long u = (unsigned long long)rd32( p ) | ( (unsigned long long)rd32( p + 4 ) << 32 );
                    double x; memcpy( &x, &u, 8 ); v = x;
                }
            }
            else switch( bytes )
            {
            case 1:  v = ( (int)p[0] - 128 ) / 128.0; break;              // 8-bit is unsigned
            case 2:  v = (short)rd16( p ) / 32768.0; break;
            case 3:  v = ( (int)( ( p[0] << 8 ) | ( p[1] << 16 ) | ( (unsigned int)p[2] << 24 ) ) >> 8 ) / 8388608.0; break;
            default: v = (int)rd32( p ) / 2147483648.0; break;
            }
            sum += v;
        }
        (*out)[(size_t)i] = (float)( sum / channels );
    }
    *out_rate = (int)rate;
    fprintf( stderr, "rt_lpc: '%s': %u ch, %u Hz, %ld frames (%.2f s), %s %u-bit\n", path, channels,
             rate, frames, (double)frames / rate, is_float ? "float" : "PCM", bytes * 8 );
    return true;
}




class WavSource : public AudioSource
{
public:
    WavSource( std::vector<float> & data ) : m_pos( 0 ) { m_data.swap( data ); }

    int read( float * out, int n )
    {
        long left = (long)m_data.size() - m_pos;
        int k = (int)std::min( (long)n, std::max( left, 0L ) );
        if( k > 0 ) memcpy( out, &m_data[(size_t)m_pos], k * sizeof(float) );
        for( int i = k; i < n; i++ ) out[i] = 0.0f;
        m_pos += k;
        return k;
    }

    long length() const { return (long)m_data.size(); }

private:
    std::vector<float> m_data;
    long m_pos;
};


AudioSource * open_wav_source( const char * path, int rate, std::string * err )
{
    std::vector<float> data;
    int file_rate = 0;
    if( !load_wav( path, &data, &file_rate, err ) ) return NULL;
    if( file_rate != rate )
    {
        fprintf( stderr, "rt_lpc: resampling %d Hz -> %d Hz\n", file_rate, rate );
        data = resample( data, file_rate, rate );
    }
    return new WavSource( data );
}




class WavSink : public AudioSink
{
public:
    WavSink( FILE * f ) : m_f( f ), m_frames( 0 ) {}
    ~WavSink() { if( m_f ) fclose( m_f ); }

    bool write( const float * in, int n )
    {
        if( !m_f ) return false;
        m_buf.resize( (size_t)n * 2 );
        for( int i = 0; i < n; i++ )
            wr16( &m_buf[(size_t)i * 2], (unsigned short)to_s16( in[i] ) );
        if( fwrite( &m_buf[0], 1, m_buf.size(), m_f ) != m_buf.size() ) return false;
        m_frames += n;

        // keep the header's sizes current
        unsigned char b[4];
        const unsigned int data_bytes = (unsigned int)( m_frames * 2 );
        fseek( m_f, 4, SEEK_SET );  wr32( b, 36 + data_bytes ); fwrite( b, 1, 4, m_f );
        fseek( m_f, 40, SEEK_SET ); wr32( b, data_bytes );      fwrite( b, 1, 4, m_f );
        fseek( m_f, 0, SEEK_END );
        return fflush( m_f ) == 0;
    }

private:
    FILE * m_f;
    long m_frames;
    std::vector<unsigned char> m_buf;
};


AudioSink * open_wav_sink( const char * path, int rate, std::string * err )
{
    FILE * f = fopen( path, "wb" );
    if( !f )
    {
        *err = std::string( "cannot create '" ) + path + "': " + strerror( errno );
        return NULL;
    }
    // 44-byte header for 16-bit mono PCM; the sizes are filled in as it grows
    unsigned char h[44];
    memcpy( h, "RIFF", 4 );      wr32( h + 4, 36 );
    memcpy( h + 8, "WAVE", 4 );
    memcpy( h + 12, "fmt ", 4 ); wr32( h + 16, 16 );
    wr16( h + 20, 1 );                      // PCM
    wr16( h + 22, 1 );                      // mono
    wr32( h + 24, (unsigned int)rate );
    wr32( h + 28, (unsigned int)rate * 2 ); // bytes per second
    wr16( h + 32, 2 );                      // bytes per frame
    wr16( h + 34, 16 );                     // bits per sample
    memcpy( h + 36, "data", 4 ); wr32( h + 40, 0 );
    if( fwrite( h, 1, 44, f ) != 44 )
    {
        fclose( f );
        *err = std::string( "cannot write '" ) + path + "'";
        return NULL;
    }
    return new WavSink( f );
}
