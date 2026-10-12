"""voice_profile.py -- voice measurements -> profile -> recommended settings.

rt_lpc --report-fd writes one JSON line per analysis frame (see rt_lpc):
    {"f0": Hz or 0, "level": dBFS, "formants": [[Hz, bandwidth], ...]}
A *profile* summarizes an utterance:
    {"f0": median pitch in Hz, "formants": [F1, F2, F3] medians in Hz,
     "frames": voiced frames used}
and a *recommendation* is the pitch and formant factors that move one
profile onto another (a target voice):
    pitch factor   = target f0 / speaker f0
    formant factor = weighted geometric mean of target Fi / speaker Fi

The target voices interviewees choose from are in a voices file
(voices.json): {"voices": [{"id", "name", "description", "f0",
"formants"}, ...]}, each the profile of a reference voice.

Formant frequencies depend on which vowels are spoken, so the speaker and the
voices must be measured reading THE SAME PHRASE. Add a voice from a recording
of a reference speaker reading it (an existing id is replaced):

    python3 voice_profile.py --rt-lpc ../path/to/rt_lpc --add-to voices.json \
        --id warm --name "Warm" --description "Low and relaxed" reference.wav

print a recording's profile:

    python3 voice_profile.py --rt-lpc ../path/to/rt_lpc speaker.wav

or the settings that would move it to each voice:

    python3 voice_profile.py --rt-lpc ../path/to/rt_lpc --compare voices.json speaker.wav

Any audio format PyAV can read works (it comes with aiortc).
"""

import json
import math
import os
import statistics
import subprocess
import sys
import threading

# a frame counts if it's voiced and no more than this far below the loudest frame
LEVEL_RANGE_DB = 30.0
# where each formant may be (Hz), and the least spacing between neighbors
FORMANT_RANGES = [(200, 1100), (600, 3200), (1400, 4300)]
MIN_SPACING = 150
# not enough speech below these
MIN_VOICED_FRAMES = 25          # about half a second of voiced speech
MIN_FORMANT_FRAMES = 15
# how much each formant counts toward the formant factor. F1 and F2 move most
# with the vowel; F3 reflects vocal tract length most steadily.
FORMANT_WEIGHTS = [0.2, 0.35, 0.45]


class NotEnoughSpeech(ValueError):
    pass


def parse_reports(lines):
    """JSON lines (str or bytes) -> list of frame dicts; skips malformed lines."""
    frames = []
    for line in lines:
        try:
            f = json.loads(line)
            frames.append({"f0": float(f["f0"]), "level": float(f["level"]),
                           "formants": [(float(a), float(b)) for a, b in f.get("formants", [])]})
        except (ValueError, KeyError, TypeError):
            continue
    return frames


def assign_formants(candidates):
    """Picks F1, F2, F3 from one frame's resonance candidates (lowest first):
    each is the lowest candidate in its range that's far enough above the
    previous one. Missing ones are None."""
    picked, floor = [], 0.0
    for lo, hi in FORMANT_RANGES:
        f = next((c for c, _ in candidates if max(lo, floor) <= c <= hi), None)
        picked.append(f)
        if f is not None:
            floor = f + MIN_SPACING
    return picked


def profile(frames):
    """Frames -> profile. Raises NotEnoughSpeech if there's too little to go on."""
    if not frames:
        raise NotEnoughSpeech("no audio was received")
    loudest = max(f["level"] for f in frames)
    voiced = [f for f in frames if f["f0"] > 0 and f["level"] >= loudest - LEVEL_RANGE_DB]
    if len(voiced) < MIN_VOICED_FRAMES:
        raise NotEnoughSpeech("not enough speech was heard; try again, a little louder")

    tracks = [[], [], []]
    for f in voiced:
        for i, v in enumerate(assign_formants(f["formants"])):
            if v is not None:
                tracks[i].append(v)
    formants = [statistics.median(t) if len(t) >= MIN_FORMANT_FRAMES else None for t in tracks]
    if formants[0] is None and formants[1] is None and formants[2] is None:
        raise NotEnoughSpeech("couldn't measure the voice clearly; try again in a quieter place")

    # median on a log scale = geometric median; robust to occasional octave errors
    f0 = math.exp(statistics.median(math.log(f["f0"]) for f in voiced))
    return {"f0": round(f0, 1),
            "formants": [round(v) if v is not None else None for v in formants],
            "frames": len(voiced)}


def recommend(speaker, standard):
    """Profile of the speaker + profile of the standard -> {"pitch", "formant"}
    factors that move the speaker toward the standard."""
    pitch = standard["f0"] / speaker["f0"]
    num = den = 0.0
    for w, s, t in zip(FORMANT_WEIGHTS, speaker["formants"], standard["formants"]):
        if s and t:
            num += w * math.log(t / s)
            den += w
    formant = math.exp(num / den) if den else 1.0
    return {"pitch": round(pitch, 3), "formant": round(formant, 3)}


def valid_profile(p) -> bool:
    return (isinstance(p, dict) and isinstance(p.get("f0"), (int, float)) and p["f0"] > 0 and
            isinstance(p.get("formants"), list) and len(p["formants"]) == 3 and
            any(isinstance(v, (int, float)) and v > 0 for v in p["formants"]))


def load_voices(path) -> list:
    """The voices file -> its list of voices, each checked: a unique id,
    a name, and a valid profile (f0 and three formants)."""
    with open(path) as fh:
        data = json.load(fh)
    voices = data.get("voices") if isinstance(data, dict) else None
    if not isinstance(voices, list) or not voices:
        raise ValueError(f"{path}: expected {{\"voices\": [{{\"id\", \"name\", \"f0\", \"formants\"}}, ...]}}")
    seen = set()
    for v in voices:
        if not (isinstance(v, dict) and isinstance(v.get("id"), str) and v["id"] and
                isinstance(v.get("name"), str) and v["name"] and valid_profile(v)):
            raise ValueError(f"{path}: each voice needs an id, a name, f0 and 3 formants: {v!r}")
        if v["id"] in seen:
            raise ValueError(f"{path}: duplicate voice id {v['id']!r}")
        seen.add(v["id"])
        v.setdefault("description", "")
    return voices


# ---------------------------------------------------------------------------
# command line: measure a recording by running it through rt_lpc
# ---------------------------------------------------------------------------
def decode_audio(path, rate):
    """Any audio file -> mono 16-bit little-endian PCM bytes at `rate`."""
    import av
    out = bytearray()
    with av.open(path) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        for frame in container.decode(audio=0):
            for f in resampler.resample(frame):
                out += bytes(f.planes[0])[: f.samples * 2]
        for f in resampler.resample(None):
            out += bytes(f.planes[0])[: f.samples * 2]
    return bytes(out)


def measure_file(rt_lpc, path, rate=24000):
    """Runs the recording through `rt_lpc --stdio --report-fd` and returns its frames."""
    pcm = decode_audio(path, rate)
    report_r, report_w = os.pipe()
    proc = subprocess.Popen([rt_lpc, "--stdio", f"--srate{rate}", f"--report-fd={report_w}"],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, pass_fds=(report_w,))
    os.close(report_w)
    lines = []

    def read_reports():
        with os.fdopen(report_r, "rb") as fh:
            lines.extend(fh)

    reader = threading.Thread(target=read_reports)
    reader.start()
    try:
        proc.stdin.write(pcm)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()
    reader.join()
    return parse_reports(lines)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Measure a voice recording with rt_lpc.")
    ap.add_argument("recording")
    ap.add_argument("--rt-lpc", default="./rt_lpc", help="path to the rt_lpc executable")
    ap.add_argument("--compare", metavar="VOICES.json",
                    help="print the settings that move this recording to each voice instead")
    ap.add_argument("--add-to", metavar="VOICES.json",
                    help="add this recording to a voices file (created if missing) as --id")
    ap.add_argument("--id", help="(--add-to) the voice's id; an existing one is replaced")
    ap.add_argument("--name", help="(--add-to) the name interviewees see")
    ap.add_argument("--description", default="", help="(--add-to) a few words about it")
    args = ap.parse_args(argv)
    if args.add_to and not (args.id and args.name):
        ap.error("--add-to needs --id and --name")
    try:
        prof = profile(measure_file(args.rt_lpc, args.recording))
    except NotEnoughSpeech as err:
        sys.exit(f"{args.recording}: {err}")

    if args.add_to:
        try:
            with open(args.add_to) as fh:
                data = json.load(fh)
        except FileNotFoundError:
            data = {"voices": []}
        voices = [v for v in data.get("voices", []) if v.get("id") != args.id]
        voices.append({"id": args.id, "name": args.name, "description": args.description,
                       "f0": prof["f0"], "formants": prof["formants"]})
        data["voices"] = voices
        data.pop("placeholder", None)
        with open(args.add_to, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        print(f"{args.add_to}: {args.id!r} saved ({len(voices)} voices)")
    elif args.compare:
        out = {"measured": prof, "voices": {v["id"]: recommend(prof, v) for v in load_voices(args.compare)}}
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(prof, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
