//-----------------------------------------------------------------------------
// name: audio_io.h
// desc: where rt_lpc's audio comes from and goes to: mono float samples
//       (+/-1.0) read from, or written to,
//         - a file descriptor, as raw signed 16-bit little-endian PCM (no
//           header). This is how a server streams audio through rt_lpc
//           over pipes; reading blocks until the samples arrive, so the
//           program runs exactly as fast as its input.
//         - a WAV file. Input files can be PCM 8/16/24/32-bit or IEEE float
//           32/64-bit (including WAVE_FORMAT_EXTENSIBLE), with any number of
//           channels (averaged to mono) at any sample rate (resampled).
//           Output files are 16-bit PCM, mono.
//-----------------------------------------------------------------------------
#ifndef __AUDIO_IO_H__
#define __AUDIO_IO_H__

#include <string>

class AudioSource
{
public:
    virtual ~AudioSource() {}
    // reads up to n samples into out. Returns how many were read: n, or
    // fewer at the end of the input (the rest of out is then zeroed), or 0
    // once there's nothing left.
    virtual int read( float * out, int n ) = 0;
    // total length in samples, or -1 if not known in advance (a stream)
    virtual long length() const { return -1; }
};

class AudioSink
{
public:
    virtual ~AudioSink() {}
    // writes n samples (clipped to +/-1). Returns false if the output is
    // gone (e.g. the reading end of a pipe closed) or on a write error.
    virtual bool write( const float * in, int n ) = 0;
};

// raw 16-bit PCM on a file descriptor (not closed by these)
AudioSource * open_pcm_source( int fd );
AudioSink * open_pcm_sink( int fd );

// a WAV file, decoded, mixed to mono and resampled to `rate` up front.
// Returns NULL with a message in *err if it can't be read.
AudioSource * open_wav_source( const char * path, int rate, std::string * err );
// a new 16-bit mono WAV file at `rate`. Its header is kept up to date after
// every write, so the file is complete even if the program is killed.
// Returns NULL with a message in *err if it can't be created.
AudioSink * open_wav_sink( const char * path, int rate, std::string * err );

#endif
