"""media_server.py -- WebRTC media service: every user's audio goes through
rt_lpc on its way to the other user.

Each browser holds ONE WebRTC connection, to this service. On it the browser
sends its microphone and receives one audio track: whatever this service
routes to it (the other caller's processed voice during a call, silence
otherwise). Changing who hears whom is just a routing change here, so calls
start and end without renegotiating WebRTC.

    caller mic --WebRTC--> [decode Opus -> 24 kHz mono PCM]
                                -> rt_lpc (stdin -> stdout)
                           [PCM -> encode Opus] --WebRTC--> callee speaker

This service is controlled only by the Node signaling server, over HTTP on
localhost (keep this port private). Endpoints:

    POST   /sessions          {"user", "sdp", "type"} -> {"sdp", "type"}
                              the user's offer; returns the answer
    DELETE /sessions/<user>   close the user's connection
    POST   /routes            {"from", "to", "process"}: send <from>'s voice to <to>,
                              through rt_lpc unless "process" is false
    DELETE /routes/<from>     stop processing <from>'s voice
    POST   /settings          {"user", "settings": {...}}: change <user>'s voice
                              settings (see SETTINGS below). Remembered for
                              the session, applied at once to a running
                              rt_lpc, and to every rt_lpc started later.

    POST   /calibration       {"user", "action": "start" | "stop" | "cancel"}:
                              measure <user>'s voice while they read the
                              calibration phrase. "stop" returns
                              {"measured": profile, "voices": [{"id", "name",
                              "description", "recommended": {"pitch", "formant"}}]}
                              (see voice_profile.py): the settings that move
                              their voice to each of the voices, or 422 {"error"}
                              if there wasn't enough speech. The measured profile
                              is kept with the session (only), for:
    POST   /voices/recommend  {"user", "voice": id} -> {"recommended": {"pitch", "formant"}}
                              the settings that move <user>'s measured voice to
                              that voice (409 if they haven't been measured)

Each rt_lpc starts with the user's current settings as command-line
options (--formant=1.2 ...). Changes while it runs reach it through a third
pipe: RT_LPC_CMD's {control_fd} is replaced with the number of a file
descriptor rt_lpc reads commands from, one per line, e.g. "formant 1.2"
(see rt_lpc's --control-fd).

Whatever rt_lpc writes to stderr is logged here, a line at a time, tagged
with the user and rt_lpc's process id. If an rt_lpc stops unexpectedly
(a crash, or an exit nobody asked for), it's logged and a new one is
started with the same settings; until it's running, that user's voice is
dropped, never sent on undisguised. Restarts back off if it keeps
failing (RESTART_DELAYS).

Configuration (environment):
    MEDIA_HOST, MEDIA_PORT  where the HTTP API listens (127.0.0.1:8081)
    RT_LPC_CMD   e.g. "./rt_lpc --stdio --srate{rate} --control-fd={control_fd}".
                 Required: without it the service won't start, so a
                 misconfiguration can never send undisguised voices. The
                 voice settings are appended as options. Without
                 {control_fd}, they can't change while it runs.
    ALLOW_UNPROCESSED=1   testing only: run without RT_LPC_CMD, passing
                 audio through unchanged.
    VOICES       the voices interviewees choose from (default: voices.json next
                 to this file). Add them with voice_profile.py --add-to, from
                 reference recordings of the calibration phrase.
    LPC_RATE     PCM rate on rt_lpc's pipes (24000)
    STUN_URLS    comma-separated STUN servers for ICE ("" for none); default
                 stun:stun.l.google.com:19302
"""

import asyncio
import fractions
import json
import logging
import os
import shlex
import signal
import time

from aiohttp import web
from aiortc import (MediaStreamTrack, RTCConfiguration, RTCIceServer,
                    RTCPeerConnection, RTCSessionDescription)
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler

import voice_profile

log = logging.getLogger("media")

RATE = int(os.environ.get("LPC_RATE", "24000"))   # PCM rate to/from rt_lpc
FRAME = RATE // 50                                # 20 ms frames
BYTES_PER_FRAME = FRAME * 2                       # mono s16
RT_LPC_CMD = os.environ.get("RT_LPC_CMD", "")
ALLOW_UNPROCESSED = os.environ.get("ALLOW_UNPROCESSED") == "1"
VOICES_FILE = os.environ.get("VOICES",
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices.json"))
CALIBRATION_MAX_SECONDS = 20                       # audio beyond this is ignored
STUN_URLS = [u for u in os.environ.get("STUN_URLS", "stun:stun.l.google.com:19302").split(",") if u]

# jitter buffer for audio on its way to a listener
TARGET_BYTES = BYTES_PER_FRAME * 2                # start playing at 40 ms buffered
MAX_BYTES = BYTES_PER_FRAME * 10                  # trim back if it reaches 200 ms
STDIN_LIMIT = BYTES_PER_FRAME * 10                # rt_lpc backlog before dropping input

# restarting an rt_lpc that stopped unexpectedly: wait this long before
# each successive attempt (the last one repeats), and start counting again
# once one has run this long
RESTART_DELAYS = (0.0, 0.5, 1.0, 2.0, 5.0)
STABLE_SECONDS = 30.0
# rt_lpc's stderr: at most this many lines per STDERR_WINDOW seconds per
# process are logged (a flood of them shouldn't drown everything else)
STDERR_MAX_LINES = 200
STDERR_WINDOW = 10.0

# Voice settings that can be sent to rt_lpc, and what values are allowed.
# These become text commands to a native program, so everything is checked
# and re-formatted here, never passed through as given. (rt_lpc checks again.)
SETTINGS = {
    "formant":    ("number", 0.5, 2.0),
    "pitch":      ("number", 0.05, 8.0),
    "order":      ("int", 1, 100),
    "gain":       ("number", -40.0, 40.0),             # dB
    "method":     ("choice", ("envelope", "poles", "lsf")),
    "excitation": ("choice", ("residual", "glottal", "impulse")),
    "emphasis":   ("bool",),
    "analysis":   ("choice", ("classic", "silk")),
}


def setting_line(name, value) -> str:
    """One validated control command, e.g. 'formant 1.2'. Raises ValueError."""
    spec = SETTINGS.get(name)
    if spec is None:
        raise ValueError(f"unknown setting {name!r}")
    kind = spec[0]
    if kind in ("number", "int"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        if not spec[1] <= value <= spec[2] or (kind == "int" and value != int(value)):
            raise ValueError(f"{name} must be {spec[1]} .. {spec[2]}")
        return f"{name} {int(value)}" if kind == "int" else f"{name} {float(value):.6g}"
    if kind == "choice":
        if value not in spec[1]:
            raise ValueError(f"{name} must be one of {', '.join(spec[1])}")
        return f"{name} {value}"
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return f"{name} {'on' if value else 'off'}"


def setting_args(settings: dict) -> list:
    """The settings as rt_lpc command-line options (e.g. --formant=1.2), so
    a new rt_lpc uses them from its very first sample: commands on the
    control pipe only take effect once its control thread has read them,
    by which time it may already have processed audio with the defaults
    (formant and pitch factors of 1: no disguise)."""
    args = []
    for name, value in settings.items():
        _, text = setting_line(name, value).split(" ", 1)     # validated and formatted
        if SETTINGS[name][0] == "bool":
            if value:
                args.append(f"--{name}")                      # off is the default
        else:
            args.append(f"--{name}={text}")
    return args


def describe_exit(code: int) -> str:
    """A process's return code, in words: "crashed (SIGSEGV)", "exited with status 1"..."""
    if code is not None and code < 0:
        try:
            return f"crashed ({signal.Signals(-code).name})"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exited with status {code}"


async def log_stderr(stream: asyncio.StreamReader, tag: str):
    """Logs a child process's stderr a line at a time as "<tag>: <line>"
    (warnings for lines that look like errors), until it closes."""
    window_start, count, suppressed = time.monotonic(), 0, 0
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode(errors="replace").rstrip()
        if not text:
            continue
        now = time.monotonic()
        if now - window_start > STDERR_WINDOW:
            if suppressed:
                log.warning("%s: (%d more lines of output not shown)", tag, suppressed)
            window_start, count, suppressed = now, 0, 0
        count += 1
        if count > STDERR_MAX_LINES:
            suppressed += 1
            continue
        lowered = text.lower()
        bad = "error" in lowered or "sanitizer" in lowered or "abort" in lowered or "fault" in lowered
        log.log(logging.WARNING if bad else logging.INFO, "%s: %s", tag, text)
    if suppressed:
        log.warning("%s: (%d more lines of output not shown)", tag, suppressed)


class OutboundTrack(MediaStreamTrack):
    """The audio a user hears: PCM pushed in by whatever is routed to them,
    paced out in real time as 20 ms frames. Silence when nothing is."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self.buf = bytearray()
        self.playing = False
        self._start = None
        self._pts = 0

    def push(self, pcm: bytes):
        self.buf += pcm
        if len(self.buf) > MAX_BYTES:                     # latency grew: skip ahead
            del self.buf[: len(self.buf) - TARGET_BYTES]

    def clear(self):
        self.buf.clear()
        self.playing = False

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError

        # pace one frame every 20 ms of wall-clock time
        if self._start is None:
            self._start = time.time()
        else:
            self._pts += FRAME
            await asyncio.sleep(self._start + self._pts / RATE - time.time())

        if not self.playing and len(self.buf) >= TARGET_BYTES:
            self.playing = True
        if self.playing and len(self.buf) >= BYTES_PER_FRAME:
            data = bytes(self.buf[:BYTES_PER_FRAME])
            del self.buf[:BYTES_PER_FRAME]
        else:
            data = bytes(BYTES_PER_FRAME)
            self.playing = False                          # ran dry: rebuffer

        frame = AudioFrame(format="s16", layout="mono", samples=FRAME)
        frame.planes[0].update(data)
        frame.sample_rate = RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, RATE)
        return frame                  # aiortc's Opus encoder resamples to 48 kHz


class Processor:
    """One direction of a call: PCM from the speaker -> rt_lpc -> the
    listener's OutboundTrack. Passthrough when process is False (or, for
    testing, when RT_LPC_CMD isn't set and ALLOW_UNPROCESSED is).

    Keeps rt_lpc running for as long as the route exists: if it stops
    without being asked to, a new one is started with the speaker's current
    settings (see _supervise()). While none is running the speaker's audio
    is dropped."""

    def __init__(self, src: str, target: OutboundTrack, settings: dict, process: bool = True):
        self.src = src
        self.target = target
        self.settings = settings      # the speaker's current settings (the session's dict)
        self.process = process
        self.passthrough = False      # audio goes straight to the listener
        self.proc = None              # the running rt_lpc, or None
        self.control = None           # write end of its control pipe
        self.tasks = []               # its stdout reader and stderr logger
        self.supervisor = None
        self.started_at = 0.0
        self.stopping = False
        self.dropped = 0              # input frames dropped: rt_lpc behind
        self.dropped_down = 0         # input frames dropped: no rt_lpc running

    async def start(self):
        if not self.process:
            log.info("%s: unprocessed", self.src)
            self.passthrough = True
            return
        if not RT_LPC_CMD:
            log.warning("%s: NOT DISGUISED (no RT_LPC_CMD; ALLOW_UNPROCESSED is set)", self.src)
            self.passthrough = True
            return
        await self._spawn()                               # OSError: the route fails
        self.supervisor = asyncio.create_task(self._supervise())

    async def _spawn(self):
        """Starts an rt_lpc with the current settings."""
        cmd = RT_LPC_CMD.replace("{rate}", str(RATE))
        pass_fds = ()
        read_end = None
        if self.control is not None:                      # the last one's control pipe
            os.close(self.control)
            self.control = None
        if "{control_fd}" in cmd:
            # a pipe for live settings; the child inherits the read end
            read_end, self.control = os.pipe()
            os.set_blocking(self.control, False)
            cmd = cmd.replace("{control_fd}", str(read_end))
            pass_fds = (read_end,)
        elif self.settings:
            log.warning("%s: RT_LPC_CMD has no {control_fd}; settings can't change until restarted", self.src)
        # the current settings from the start; later changes go down the pipe
        argv = shlex.split(cmd) + setting_args(self.settings)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, pass_fds=pass_fds)
        except OSError:
            if self.control is not None:
                os.close(self.control)
                self.control = None
            raise
        finally:
            if read_end is not None:
                os.close(read_end)                        # the child has its own copy now
        self.proc = proc
        self.started_at = time.monotonic()
        log.info("%s: started %s (pid %d)", self.src, " ".join(argv), proc.pid)
        self.tasks = [asyncio.create_task(self._read(proc)),
                      asyncio.create_task(log_stderr(proc.stderr, f"{self.src}: rt_lpc[{proc.pid}]"))]

    async def _supervise(self):
        """Waits for rt_lpc to exit; unless that was asked for, starts
        another with the same (current) settings, backing off if they keep
        failing."""
        failures = 0
        while True:
            proc = self.proc
            code = await proc.wait()
            # let its last output and messages through first
            await asyncio.gather(*self.tasks, return_exceptions=True)
            if self.stopping:
                return
            self.proc = None                              # input is dropped from here
            if time.monotonic() - self.started_at >= STABLE_SECONDS:
                failures = 0
            log.error("%s: rt_lpc (pid %d) %s; restarting it with the same settings",
                      self.src, proc.pid, describe_exit(code))
            while True:
                delay = RESTART_DELAYS[min(failures, len(RESTART_DELAYS) - 1)]
                failures += 1
                if delay:
                    await asyncio.sleep(delay)
                if self.stopping:
                    return
                try:
                    await self._spawn()
                    break
                except OSError as err:
                    log.error("%s: cannot restart rt_lpc: %s", self.src, err)
            if self.dropped_down:
                log.info("%s: %d input frames dropped while rt_lpc was down", self.src, self.dropped_down)
                self.dropped_down = 0

    def send_settings(self, settings: dict):
        """Writes control commands for these (already validated) settings.
        (While rt_lpc is down there's nowhere to send them; the restart
        applies all of the current settings.)"""
        if self.control is None or not settings:
            return
        data = "".join(setting_line(k, v) + "\n" for k, v in settings.items()).encode()
        try:
            os.write(self.control, data)                  # a few dozen bytes: fits the pipe
        except BlockingIOError:
            log.warning("%s: rt_lpc isn't reading its control pipe; settings dropped", self.src)
        except (BrokenPipeError, OSError):
            pass                                          # rt_lpc has exited

    def feed(self, pcm: bytes):
        if self.passthrough:
            self.target.push(pcm)
            return
        proc = self.proc
        if proc is None or proc.returncode is not None or proc.stdin.is_closing():
            self.dropped_down += 1                        # never send it on unprocessed
            return
        # rt_lpc not keeping up: drop input instead of accumulating delay
        if proc.stdin.transport.get_write_buffer_size() > STDIN_LIMIT:
            self.dropped += 1
            return
        proc.stdin.write(pcm)

    async def _read(self, proc):
        carry = b""
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            chunk = carry + chunk
            cut = len(chunk) & ~1                         # whole 16-bit samples only
            carry = chunk[cut:]
            self.target.push(chunk[:cut])

    async def stop(self):
        self.stopping = True
        if self.control is not None:
            os.close(self.control)                        # control thread sees end of file
            self.control = None
        proc = self.proc
        if proc is not None:
            try:
                proc.stdin.close()                        # end of stream: rt_lpc exits
                code = await asyncio.wait_for(proc.wait(), 1.0)
                if code != 0:
                    log.warning("%s: rt_lpc (pid %d) %s while stopping", self.src, proc.pid, describe_exit(code))
            except asyncio.TimeoutError:
                proc.kill()
            except ProcessLookupError:
                pass
        if self.supervisor:
            self.supervisor.cancel()
        if self.tasks:
            await asyncio.wait(self.tasks, timeout=1.0)   # the last stderr lines
            for t in self.tasks:
                t.cancel()
        if self.dropped:
            log.info("%s: dropped %d input frames", self.src, self.dropped)


class CalibrationFailed(Exception):
    pass


class Calibrator:
    """Measures a user's voice: their mic audio goes through a separate
    rt_lpc that does nothing but report its measurements (--report-fd) while
    they read the calibration phrase. Its audio output is thrown away."""

    def __init__(self, user: str):
        self.user = user
        self.proc = None
        self.collector = None
        self.lines = []
        self.started = None
        self.stderr_task = None

    async def start(self):
        cmd = RT_LPC_CMD.replace("{rate}", str(RATE))
        pass_fds, extra = [], []
        if "{control_fd}" in cmd:
            # nothing to control: give it a pipe that's already at end of file
            ctl_r, ctl_w = os.pipe()
            os.close(ctl_w)
            cmd = cmd.replace("{control_fd}", str(ctl_r))
            pass_fds.append(ctl_r)
        report_r, report_w = os.pipe()
        pass_fds.append(report_w)
        argv = shlex.split(cmd) + [f"--report-fd={report_w}"]
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE, pass_fds=pass_fds)
        except OSError:
            os.close(report_r)
            raise
        finally:
            for fd in pass_fds:
                os.close(fd)                              # the child has its own copies
        reader = asyncio.StreamReader()
        await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(report_r, "rb", buffering=0))
        self.collector = asyncio.create_task(self._collect(reader))
        self.stderr_task = asyncio.create_task(
            log_stderr(self.proc.stderr, f"{self.user}: rt_lpc[{self.proc.pid}] (calibration)"))
        self.started = time.time()
        log.info("%s: calibration started (pid %d)", self.user, self.proc.pid)

    async def _collect(self, reader):
        while True:
            line = await reader.readline()
            if not line:
                break
            self.lines.append(line)

    def feed(self, pcm: bytes):
        if self.proc is None or self.proc.stdin.is_closing():
            return
        if time.time() - self.started > CALIBRATION_MAX_SECONDS:
            return
        if self.proc.stdin.transport.get_write_buffer_size() > STDIN_LIMIT:
            return
        self.proc.stdin.write(pcm)

    async def finish(self) -> dict:
        """Stops measuring; returns the measured profile. Raises
        voice_profile.NotEnoughSpeech if there wasn't enough to go on, or
        CalibrationFailed if rt_lpc stopped unexpectedly."""
        crashed_early = self.proc.returncode not in (None, 0)
        if not self.proc.stdin.is_closing():
            self.proc.stdin.close()                       # rt_lpc finishes and exits
        try:
            code = await asyncio.wait_for(self.proc.wait(), 3.0)
        except asyncio.TimeoutError:
            self.proc.kill()
            code = await self.proc.wait()
        try:
            await asyncio.wait_for(self.collector, 2.0)
        except asyncio.TimeoutError:
            self.collector.cancel()
        if self.stderr_task:
            await asyncio.wait([self.stderr_task], timeout=1.0)
        if crashed_early or code != 0:
            self.lines = []
            log.error("%s: calibration's rt_lpc (pid %d) %s", self.user, self.proc.pid, describe_exit(code))
            raise CalibrationFailed("the voice measurement stopped unexpectedly; please try again")
        frames = voice_profile.parse_reports(self.lines)
        self.lines = []                                   # don't keep voice data around
        return voice_profile.profile(frames)

    async def cancel(self):
        if self.proc and self.proc.returncode is None:
            self.proc.kill()
        if self.collector:
            self.collector.cancel()
        self.lines = []


class Session:
    """One user's WebRTC connection to this service."""

    def __init__(self, user: str):
        self.user = user
        config = RTCConfiguration(iceServers=[RTCIceServer(urls=u) for u in STUN_URLS])
        self.pc = RTCPeerConnection(configuration=config)
        self.out = OutboundTrack()
        self.route = None             # Processor for this user's voice, if routed
        self.pump = None
        self.settings = {}            # voice settings for this user's rt_lpc processes
        self.calibrator = None        # Calibrator while measuring this user's voice
        self.measured = None          # their voice's profile, once measured (kept only here)

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                self.pump = asyncio.create_task(self._pump(track))

        @self.pc.on("connectionstatechange")
        async def on_state():
            log.info("%s: connection %s", user, self.pc.connectionState)

    async def _pump(self, track):
        """Mic frames from the browser -> 24 kHz mono PCM -> current route.
        Always reads, so the receive side never backs up."""
        resampler = AudioResampler(format="s16", layout="mono", rate=RATE)
        pending = bytearray()
        try:
            while True:
                frame = await track.recv()
                for f in resampler.resample(frame):
                    pending += bytes(f.planes[0])[: f.samples * 2]
                while len(pending) >= BYTES_PER_FRAME:
                    chunk = bytes(pending[:BYTES_PER_FRAME])
                    del pending[:BYTES_PER_FRAME]
                    if self.route:
                        self.route.feed(chunk)
                    if self.calibrator:
                        self.calibrator.feed(chunk)
        except MediaStreamError:
            pass

    async def answer(self, sdp: str, type_: str) -> RTCSessionDescription:
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
        # the one track this user will hear, sent back on the same transceiver
        for t in self.pc.getTransceivers():
            if t.kind == "audio":
                t.sender.replaceTrack(self.out)
                t.direction = "sendrecv"
                break
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        return self.pc.localDescription           # aiortc includes all candidates

    async def unroute(self):
        if self.route:
            route, self.route = self.route, None
            route.target.clear()
            await route.stop()

    async def close(self):
        if self.calibrator:
            await self.calibrator.cancel()
            self.calibrator = None
        await self.unroute()
        if self.pump:
            self.pump.cancel()
        await self.pc.close()


sessions: dict[str, Session] = {}
VOICES = []                   # the voices to choose from, loaded in main()


def voice_choices(measured: dict) -> list:
    """Every voice, with the settings that move this voice to it."""
    return [{"id": v["id"], "name": v["name"], "description": v["description"],
             "recommended": voice_profile.recommend(measured, v)} for v in VOICES]


async def post_session(request):
    body = await request.json()
    user = body["user"]
    measured = None
    if user in sessions:                         # reconnecting: replace the old one,
        measured = sessions[user].measured       # keeping their measured voice
        await remove(user)
    session = sessions[user] = Session(user)
    session.measured = measured
    desc = await session.answer(body["sdp"], body["type"])
    log.info("%s: session created", user)
    return web.json_response({"sdp": desc.sdp, "type": desc.type})


async def remove(user):
    session = sessions.pop(user, None)
    if session is None:
        return
    # anything routed TO this user goes nowhere now. (A copy: unroute()
    # awaits, and the other user may be removed meanwhile.)
    for other in list(sessions.values()):
        if other.route and other.route.target is session.out:
            await other.unroute()
    await session.close()
    log.info("%s: session closed", user)


async def delete_session(request):
    await remove(request.match_info["user"])
    return web.json_response({"ok": True})


async def post_route(request):
    body = await request.json()
    src, dst = sessions.get(body["from"]), sessions.get(body["to"])
    if src is None or dst is None:
        return web.json_response({"error": "unknown user"}, status=404)
    await src.unroute()
    # processed unless explicitly told otherwise
    processor = Processor(src.user, dst.out, src.settings, process=body.get("process") is not False)
    try:
        await processor.start()
    except OSError as err:
        log.error("%s: cannot start rt_lpc: %s", src.user, err)
        return web.json_response({"error": f"cannot start rt_lpc: {err}"}, status=500)
    src.route = processor
    log.info("route %s -> %s", src.user, dst.user)
    return web.json_response({"ok": True})


async def delete_route(request):
    session = sessions.get(request.match_info["user"])
    if session:
        await session.unroute()
    return web.json_response({"ok": True})


async def post_settings(request):
    body = await request.json()
    session = sessions.get(body.get("user"))
    if session is None:
        return web.json_response({"error": "unknown user"}, status=404)
    changes = body.get("settings")
    if not isinstance(changes, dict):
        return web.json_response({"error": "settings must be an object"}, status=400)
    try:
        for name, value in changes.items():
            setting_line(name, value)                     # validate everything first
    except ValueError as err:
        return web.json_response({"error": str(err)}, status=400)
    session.settings.update(changes)
    if session.route:
        session.route.send_settings(changes)
    return web.json_response({"ok": True, "settings": session.settings})


async def post_calibration(request):
    body = await request.json()
    session = sessions.get(body.get("user"))
    if session is None:
        return web.json_response({"error": "unknown user"}, status=404)
    action = body.get("action")
    if action == "start":
        if not VOICES or not RT_LPC_CMD:
            return web.json_response({"error": "voice calibration isn't set up on this server"}, status=503)
        if session.calibrator:
            await session.calibrator.cancel()
        calibrator = Calibrator(session.user)
        try:
            await calibrator.start()
        except OSError as err:
            return web.json_response({"error": f"cannot start rt_lpc: {err}"}, status=500)
        session.calibrator = calibrator
        return web.json_response({"ok": True, "maxSeconds": CALIBRATION_MAX_SECONDS})
    if action == "stop":
        calibrator, session.calibrator = session.calibrator, None
        if calibrator is None:
            return web.json_response({"error": "not calibrating"}, status=409)
        try:
            measured = await calibrator.finish()
        except voice_profile.NotEnoughSpeech as err:
            return web.json_response({"error": str(err)}, status=422)
        except CalibrationFailed as err:
            return web.json_response({"error": str(err)}, status=500)
        session.measured = measured
        log.info("%s: calibration finished", session.user)   # (not the measurements)
        return web.json_response({"measured": measured, "voices": voice_choices(measured)})
    if action == "cancel":
        if session.calibrator:
            await session.calibrator.cancel()
            session.calibrator = None
        return web.json_response({"ok": True})
    return web.json_response({"error": "action must be start, stop or cancel"}, status=400)


async def post_voice_recommend(request):
    body = await request.json()
    session = sessions.get(body.get("user"))
    if session is None:
        return web.json_response({"error": "unknown user"}, status=404)
    voice = next((v for v in VOICES if v["id"] == body.get("voice")), None)
    if voice is None:
        return web.json_response({"error": "unknown voice"}, status=404)
    if session.measured is None:
        return web.json_response({"error": "record the phrase first"}, status=409)
    return web.json_response({"recommended": voice_profile.recommend(session.measured, voice)})


async def on_shutdown(app):
    for user in list(sessions):
        await remove(user)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not RT_LPC_CMD and not ALLOW_UNPROCESSED:
        log.error("RT_LPC_CMD is not set, so voices would not be disguised. Set it, or set "
                  "ALLOW_UNPROCESSED=1 for testing without rt_lpc.")
        raise SystemExit(1)
    logging.getLogger("aioice").setLevel(logging.WARNING)
    app = web.Application()
    app.add_routes([
        web.post("/sessions", post_session),
        web.delete("/sessions/{user}", delete_session),
        web.post("/routes", post_route),
        web.delete("/routes/{user}", delete_route),
        web.post("/settings", post_settings),
        web.post("/calibration", post_calibration),
        web.post("/voices/recommend", post_voice_recommend),
    ])
    app.on_shutdown.append(on_shutdown)
    global VOICES
    try:
        VOICES = voice_profile.load_voices(VOICES_FILE)
        log.info("voices: %s", ", ".join(v["name"] for v in VOICES))
        with open(VOICES_FILE) as fh:
            if json.load(fh).get("placeholder"):
                log.warning("%s holds placeholder voices; replace them with recordings of reference "
                            "speakers reading the calibration phrase (voice_profile.py --add-to)", VOICES_FILE)
    except (OSError, ValueError) as err:
        log.warning("voice calibration disabled: %s", err)
    log.info("rt_lpc: %s", RT_LPC_CMD or "(not set: NO DISGUISE, testing only)")
    web.run_app(app, host=os.environ.get("MEDIA_HOST", "127.0.0.1"),
                port=int(os.environ.get("MEDIA_PORT", "8081")), print=None,
                access_log=None)


if __name__ == "__main__":
    main()
