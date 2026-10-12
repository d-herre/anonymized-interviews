// server.js -- static files + signaling for anonymized interviews.
//
// Two kinds of users:
//   interviewee   opens /           Sets up their voice disguise, joins a queue
//                                   when ready, waits (and can still adjust it),
//                                   then is interviewed. Their
//                                   voice always goes through rt_lpc; they may
//                                   share their screen.
//   interviewer   opens /interviewer   Needs the access key. Takes the next
//                                   interviewee from the queue, which starts
//                                   the call. Their own voice is not changed.
//
// Audio never touches this process: each browser's WebRTC connection goes to
// the Python media service (media/media_server.py), which this server controls
// over HTTP on localhost. Screen sharing goes directly between the two
// browsers; only its setup messages pass through here.
//
//   python3 media/media_server.py          (with RT_LPC_CMD set; see there)
//   INTERVIEWER_KEY=... node server.js     (without it, a random key is printed)
//
// Browsers only allow microphones and screen capture on https:// or
// http://localhost, so put this behind HTTPS when deploying.
//
// Messages, browser -> server (JSON). The first must be 'hello':
//   {type:'hello', role:'interviewee'}  or  {type:'hello', role:'interviewer', key}
//   {type:'offer', sdp}        WebRTC offer for the media service
//   interviewee only (once their audio is up):
//     {type:'join'}            join the queue, when they're ready (again after an
//                              interview or after leaving it)
//     {type:'leave'}           leave the queue
//     {type:'voice', settings} change the voice disguise, e.g. {formant: 1.2}
//     {type:'monitor', on}     hear your own disguised voice (while not in an interview)
//     {type:'calibrate', action: 'start' | 'stop' | 'cancel'}
//                              measure your voice while you read the calibration
//                              phrase; 'stop' gets you the voice options, with
//                              one suggested (and applied)
//     {type:'voiceOption', id} switch to one of the voice options (after
//                              calibrating; the settings are worked out again
//                              from your measured voice)
//   interviewer only:
//     {type:'next'}            take the next interviewee and start the interview
//   both:
//     {type:'hangup'}          end the interview
//     {type:'screen', role, sdp | candidate | stop}
//                              screen-sharing setup, relayed to the other person.
//                              Only the interviewee shares; only the interviewer views.
// server -> browser:
//   {type:'welcome', id, role, iceServers, voice?, voiceSpec?, disguise?}
//   {type:'answer', sdp}
//   {type:'queue', position?, length}   interviewees get their place (1 = next)
//   {type:'connected', peer} {type:'ended', peer, reason, requeued?}
//   {type:'voice', voice, option}  the disguise as applied (after clamping), and
//                              the voice option it came from (null once the
//                              pitch or formant factor is changed by hand)
//   {type:'calibration', state: 'recording' | 'analyzing' | 'done' | 'failed' | 'cancelled',
//      maxSeconds?, measured?, voices?, suggested?, message?}
//                              voices: [{id, name, description, settings:{pitch, formant},
//                              adjusted, tags}], tags from 'suggested', 'close'
//                              (barely different from your voice, so the pitch
//                              is pushed further), 'strained' (may be hard to
//                              understand)
//   {type:'voiceOption', id, settings, adjusted}   a voice option was applied
//   {type:'recalibrate', message}   your measurement was lost: record the phrase again
//   {type:'screen', from, role, sdp | candidate | stop}
//   {type:'error', message}

import express from 'express';
import http from 'node:http';
import { randomBytes, createHash, timingSafeEqual } from 'node:crypto';
import { WebSocketServer } from 'ws';

const PORT = Number(process.env.PORT || 9000);
const MEDIA_URL = process.env.MEDIA_URL || 'http://127.0.0.1:8081';
// what the browser uses for ICE; the media service has its own setting
const ICE_SERVERS = (process.env.STUN_URLS ?? 'stun:stun.l.google.com:19302')
  .split(',').filter(Boolean).map((urls) => ({ urls }));

// ---- interviewer access ----
// Anyone with the key can take interviewees from the queue and hear them, so
// treat it like a password. Without INTERVIEWER_KEY, a random one is made at
// startup and printed.
const INTERVIEWER_KEY = process.env.INTERVIEWER_KEY || randomBytes(12).toString('base64url');
const keyDigest = (k) => createHash('sha256').update(String(k)).digest();
const KEY_DIGEST = keyDigest(INTERVIEWER_KEY);
const keyMatches = (k) => typeof k === 'string' && timingSafeEqual(keyDigest(k), KEY_DIGEST);

// ---- voice disguise (interviewees) ----
// Each interviewee controls their own disguise. Values from the browser are
// checked against this and clamped before they go anywhere. The browser gets
// this too, to build its controls.
const VOICE_SPEC = {
  formant:    { type: 'number', min: 0.5,  max: 2.0, step: 0.01, label: 'Formant factor' },
  pitch:      { type: 'number', min: 0.25, max: 4.0, step: 0.01, label: 'Pitch factor' },
  gain:       { type: 'number', min: -20,  max: 30,  step: 1,    label: 'Volume (dB)' },
  order:      { type: 'int',    min: 8,    max: 60,  step: 1,    label: 'LPC order' },
  method:     { type: 'choice', options: ['envelope', 'poles', 'lsf'], label: 'Formant method' },
  excitation: { type: 'choice', options: ['residual', 'glottal', 'impulse'], label: 'Excitation' },
  emphasis:   { type: 'bool', label: 'Pre-/de-emphasis' },
  analysis:   { type: 'choice', options: ['classic', 'silk'], label: 'Analysis' },
};
// Starts disguised: formants and pitch both moved. The residual excitation
// (the interviewee's own voice, pitch-shifted, through the formant-shifted
// filter) sounds the most natural, and rt_lpc keeps its loudness matched to
// the input, so no gain is needed. (The glottal and impulse excitations come
// out about 5 dB quieter.)
const DEFAULT_VOICE = {
  formant: 1.15, pitch: 1.3, gain: 0, order: 30,
  method: 'envelope', excitation: 'residual', emphasis: false, analysis: 'classic',
};
// The disguise can't be turned off: at least one of formant and pitch must be
// moved by at least this factor (in either direction).
const DISGUISE = { formant: 1.08, pitch: 1.15 };
const farEnough = (v, f) => Math.max(v, 1 / v) >= f;
const isDisguised = (voice) => farEnough(voice.formant, DISGUISE.formant) || farEnough(voice.pitch, DISGUISE.pitch);
// ---- voice calibration and voice options ----
// The interviewee reads this aloud once; their pitch and formants are measured
// and compared with each voice option (media/voices.json: reference speakers
// reading THE SAME phrase, since formants depend on the vowels). Moving toward
// an option gives pitch and formant factors for it.
const CALIBRATION_PHRASE = 'My mother and I were away all day on Monday, so we hope you are ready now.';
const CALIBRATION_SECONDS = Number(process.env.CALIBRATION_SECONDS || 15);   // stop measuring automatically after this

// During an interview only these may change: hearing the same person through
// two different disguises would make them easier to identify.
const CHANGEABLE_IN_INTERVIEW = new Set(['gain']);

// returns only the valid fields of `input`, clamped; throws on a bad value
function sanitizeVoice(input) {
  if (!input || typeof input !== 'object') throw new Error('settings must be an object');
  const out = {};
  for (const [name, value] of Object.entries(input)) {
    const spec = VOICE_SPEC[name];
    if (!spec) throw new Error(`unknown setting ${name}`);
    if (spec.type === 'number' || spec.type === 'int') {
      let v = Number(value);
      if (typeof value === 'boolean' || !Number.isFinite(v)) throw new Error(`${name} must be a number`);
      v = Math.min(spec.max, Math.max(spec.min, v));
      out[name] = spec.type === 'int' ? Math.round(v) : Math.round(v * 1000) / 1000;
    } else if (spec.type === 'choice') {
      if (!spec.options.includes(value)) throw new Error(`${name} must be one of ${spec.options.join(', ')}`);
      out[name] = value;
    } else {
      out[name] = Boolean(value);
    }
  }
  return out;
}

// Measured recommendation -> settings this interviewee may use: clamped to the
// allowed ranges and, if that would leave the voice undisguised (their voice
// is already close to the option's), with the pitch pushed just past the
// disguise threshold in the direction it was going.
function recommendedSettings(rec, voice) {
  const out = sanitizeVoice({ pitch: rec.pitch, formant: rec.formant });
  let adjusted = out.pitch !== rec.pitch || out.formant !== rec.formant;
  if (!isDisguised({ ...voice, ...out })) {
    const past = DISGUISE.pitch * 1.01;    // a little past the minimum, either way
    out.pitch = Math.round((rec.pitch >= 1 ? past : 1 / past) * 1000) / 1000;
    adjusted = true;
  }
  return { settings: out, adjusted };
}

// Ranking the options for one interviewee, from the factors that would move
// their voice to each one:
//   strength  how far the disguise moves it, in units of the DISGUISE minimum
//             (root sum of squares of the two factors' log-distances from 1;
//             1 = just disguised)
//   strain    how near the factors come to the limits past which the voice
//             gets hard to understand (the larger of the two; 1 = at a limit)
// The suggestion is the gentlest option that still disguises clearly: the
// least strain among those with strength >= SUGGEST_STRENGTH and strain <= 1;
// failing that, the strongest with strain <= 1; failing that, the least strain.
const INTELLIGIBLE = { formant: 1.25, pitch: 1.8 };
const SUGGEST_STRENGTH = 1.5;
const logRatio = (v, f) => Math.abs(Math.log(v)) / Math.log(f);
const strengthOf = (r) => Math.hypot(logRatio(r.pitch, DISGUISE.pitch), logRatio(r.formant, DISGUISE.formant));
const strainOf = (r) => Math.max(logRatio(r.pitch, INTELLIGIBLE.pitch), logRatio(r.formant, INTELLIGIBLE.formant));

// media service's {voices: [{id, name, description, recommended}]} -> the
// options as sent to the browser, and the id of the one to suggest
function rankVoices(voices, voice) {
  const options = voices.map((v) => {
    const { settings, adjusted } = recommendedSettings(v.recommended, voice);
    const strength = strengthOf(v.recommended), strain = strainOf(v.recommended);
    const tags = [];
    if (strength < 1) tags.push('close');
    if (strain > 1) tags.push('strained');
    return { id: v.id, name: v.name, description: v.description, settings, adjusted, tags, strength, strain };
  });
  const usable = options.filter((o) => o.strain <= 1);
  const by = (key, sign) => (a, b) => sign * (a[key] - b[key]);
  const pick = usable.filter((o) => o.strength >= SUGGEST_STRENGTH).sort(by('strain', 1))[0]
    ?? usable.sort(by('strength', -1))[0]
    ?? [...options].sort(by('strain', 1))[0];
  if (pick) pick.tags.unshift('suggested');
  return {
    options: options.map(({ strength, strain, ...o }) => o),
    suggested: pick?.id ?? null,
  };
}

const app = express();
const PUBLIC = new URL('./public', import.meta.url).pathname;
app.get('/interviewer', (req, res) => res.sendFile('interviewer.html', { root: PUBLIC }));
app.use(express.static(PUBLIC));
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/signal', maxPayload: 256 * 1024 });

// ---- media service API ----
async function media(method, path, body) {
  const res = await fetch(MEDIA_URL + path, {
    method,
    headers: body ? { 'content-type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `media service: HTTP ${res.status}`);
  return data;
}

// ---- users and the queue ----
// id -> { id, ws, role, mediaReady, peer, monitor, voice }
const users = new Map();
const queue = [];                 // interviewee ids, first = next

function newId() {
  let id;
  do id = randomBytes(4).toString('hex'); while (users.has(id));
  return id;
}

function send(user, msg) {
  if (user && user.ws.readyState === user.ws.OPEN) user.ws.send(JSON.stringify(msg));
}

function removeFromQueue(user) {
  const i = queue.indexOf(user.id);
  if (i >= 0) queue.splice(i, 1);
  return i >= 0;
}

// everyone learns the queue length; interviewees also their place in it
function broadcastQueue() {
  for (const u of users.values()) {
    if (u.role === 'interviewer') send(u, { type: 'queue', length: queue.length });
    else if (u.role === 'interviewee') {
      const i = queue.indexOf(u.id);
      send(u, { type: 'queue', position: i >= 0 ? i + 1 : null, length: queue.length });
    }
  }
}

// Where this user's voice goes: to the other person in an interview, back to
// themselves when an interviewee is trying out their disguise, nowhere
// otherwise. Interviewee audio is ALWAYS processed by rt_lpc; the
// interviewer's never is.
async function updateRoute(user) {
  if (!user.mediaReady) return;
  const to = user.peer ?? (user.role === 'interviewee' && user.monitor ? user.id : null);
  if (to) await media('POST', '/routes', { from: user.id, to, process: user.role === 'interviewee' });
  else await media('DELETE', `/routes/${user.id}`);
}

// ends the user's interview, if any; returns the other person, who is told
// why (plus anything in `extra`)
async function endInterview(user, reason, extra = {}) {
  const peer = users.get(user.peer);
  user.peer = null;
  if (peer) {
    peer.peer = null;
    send(peer, { type: 'ended', peer: user.id, reason, ...extra });
    await updateRoute(peer).catch(() => {});
  }
  await updateRoute(user).catch(() => {});
  return peer;
}

// Measured values go only to the interviewee they belong to; they aren't
// logged or kept here.
async function finishCalibration(user) {
  clearTimeout(user.calibrationTimer);
  user.calibrating = false;
  send(user, { type: 'calibration', state: 'analyzing' });
  let result;
  try {
    result = await media('POST', '/calibration', { user: user.id, action: 'stop' });
  } catch (err) {
    send(user, { type: 'calibration', state: 'failed', message: err.message });
    return;
  }
  const { options, suggested } = rankVoices(result.voices ?? [], user.voice);
  user.voiceOptions = new Set(options.map((o) => o.id));
  send(user, {
    type: 'calibration', state: 'done',
    measured: { f0: result.measured.f0, formants: result.measured.formants },
    voices: options, suggested,
  });
  // start them off with the suggestion (the interview may have begun meanwhile)
  if (suggested && !user.peer) await chooseVoiceOption(user, suggested).catch((err) => {
    send(user, { type: 'error', message: err.message });
  });
}

// Validated, clamped changes -> applied to this interviewee's voice, if they
// may make them now. `option`: the voice option they came from, or undefined
// for a change by hand (which, if it touches pitch or formant, means the
// voice no longer is that option).
async function applyVoice(user, changes, option) {
  if (user.peer) {
    const locked = Object.keys(changes).filter((k) => !CHANGEABLE_IN_INTERVIEW.has(k));
    if (locked.length) {
      send(user, { type: 'voice', voice: user.voice, option: user.voiceOption });   // put the controls back
      throw new Error('only the volume can be changed during the interview');
    }
  }
  const next = { ...user.voice, ...changes };
  if (!isDisguised(next)) {
    send(user, { type: 'voice', voice: user.voice, option: user.voiceOption });
    throw new Error('your voice must stay disguised: move the formant or pitch factor further from 1');
  }
  user.voice = next;
  if (option !== undefined) user.voiceOption = option;
  else if ('pitch' in changes || 'formant' in changes) user.voiceOption = null;
  if (user.mediaReady && Object.keys(changes).length)
    await media('POST', '/settings', { user: user.id, settings: changes });
  send(user, { type: 'voice', voice: user.voice, option: user.voiceOption });
}

// Works out this interviewee's settings for voice option `id` from their
// measured voice (kept by the media service) and applies them.
async function chooseVoiceOption(user, id) {
  if (user.peer) throw new Error('the voice can\'t be changed during the interview');
  let result;
  try {
    result = await media('POST', '/voices/recommend', { user: user.id, voice: id });
  } catch (err) {
    if (/record the phrase/.test(err.message)) {
      user.voiceOptions = null;
      send(user, { type: 'recalibrate', message: 'Your voice measurement was lost. Please read the phrase again.' });
      return;
    }
    throw err;
  }
  const { settings, adjusted } = recommendedSettings(result.recommended, user.voice);
  await applyVoice(user, settings, id);
  send(user, { type: 'voiceOption', id, settings, adjusted });
}

async function cancelCalibration(user) {
  clearTimeout(user.calibrationTimer);
  if (!user.calibrating) return;
  user.calibrating = false;
  await media('POST', '/calibration', { user: user.id, action: 'cancel' }).catch(() => {});
}

function requireRole(user, role) {
  if (user.role !== role) throw new Error(`only ${role}s can do that`);
}

const handlers = {
  async offer(user, msg) {
    const answer = await media('POST', '/sessions', { user: user.id, sdp: msg.sdp, type: 'offer' });
    // the new media session starts with this user's voice settings
    if (user.voice) await media('POST', '/settings', { user: user.id, settings: user.voice });
    user.mediaReady = true;
    send(user, { type: 'answer', sdp: answer.sdp });
    // after a reconnect, restore routing in both directions (the media service
    // dropped any route into the old connection)
    await updateRoute(user);
    if (user.peer) await updateRoute(users.get(user.peer));
  },

  // ---- interviewee ----
  async join(user) {
    requireRole(user, 'interviewee');
    if (!user.mediaReady) throw new Error('audio not started');
    if (user.peer) throw new Error('already in an interview');
    if (!queue.includes(user.id)) queue.push(user.id);
    broadcastQueue();
  },

  async leave(user) {
    requireRole(user, 'interviewee');
    if (removeFromQueue(user)) broadcastQueue();
  },

  async voice(user, msg) {
    requireRole(user, 'interviewee');
    await applyVoice(user, sanitizeVoice(msg.settings));
  },

  async voiceOption(user, msg) {
    requireRole(user, 'interviewee');
    if (!user.mediaReady) throw new Error('audio not started');
    if (!user.voiceOptions) throw new Error('read the calibration phrase first');
    if (typeof msg.id !== 'string' || !user.voiceOptions.has(msg.id)) throw new Error('unknown voice option');
    await chooseVoiceOption(user, msg.id);
  },

  async calibrate(user, msg) {
    requireRole(user, 'interviewee');
    if (!user.mediaReady) throw new Error('audio not started');
    if (msg.action === 'start') {
      if (user.peer) throw new Error('not during the interview');
      clearTimeout(user.calibrationTimer);
      const r = await media('POST', '/calibration', { user: user.id, action: 'start' });
      user.calibrating = true;
      const seconds = Math.min(CALIBRATION_SECONDS, r.maxSeconds ?? CALIBRATION_SECONDS);
      // stop by itself if the interviewee never presses Done
      user.calibrationTimer = setTimeout(() => {
        user.pending = user.pending.then(() => user.calibrating && finishCalibration(user));
      }, seconds * 1000);
      send(user, { type: 'calibration', state: 'recording', maxSeconds: seconds });
    } else if (msg.action === 'stop') {
      if (!user.calibrating) throw new Error('not measuring');
      await finishCalibration(user);
    } else if (msg.action === 'cancel') {
      await cancelCalibration(user);
      send(user, { type: 'calibration', state: 'cancelled' });
    } else {
      throw new Error('bad calibrate message');
    }
  },

  async monitor(user, msg) {
    requireRole(user, 'interviewee');
    user.monitor = !!msg.on;
    if (!user.peer) await updateRoute(user);
  },

  // ---- interviewer ----
  async next(user) {
    requireRole(user, 'interviewer');
    if (!user.mediaReady) throw new Error('audio not started');
    if (user.peer) throw new Error('already in an interview');

    let interviewee = null;
    while (queue.length && !interviewee) {
      const u = users.get(queue.shift());
      if (u && u.mediaReady && !u.peer) interviewee = u;
    }
    broadcastQueue();
    if (!interviewee) throw new Error('nobody is waiting');

    interviewee.monitor = false;
    if (interviewee.calibrating) {
      await cancelCalibration(interviewee);
      send(interviewee, { type: 'calibration', state: 'cancelled' });
    }
    interviewee.peer = user.id;
    user.peer = interviewee.id;
    try {
      // interviewee -> rt_lpc -> interviewer; interviewer -> interviewee unchanged
      await Promise.all([updateRoute(interviewee), updateRoute(user)]);
    } catch (err) {
      // couldn't set up the audio: undo, and put them back at the front
      interviewee.peer = null;
      user.peer = null;
      await Promise.all([updateRoute(interviewee), updateRoute(user)]).catch(() => {});
      queue.unshift(interviewee.id);
      broadcastQueue();
      throw new Error(`could not start the interview: ${err.message}`);
    }
    send(user, { type: 'connected', peer: interviewee.id });
    send(interviewee, { type: 'connected', peer: user.id });
  },

  // ---- both ----
  async hangup(user) {
    const peer = await endInterview(user, 'hung up');
    if (peer) send(user, { type: 'ended', peer: peer.id, reason: 'you hung up' });
  },

  // Screen sharing is peer-to-peer: only its setup messages pass through
  // here, only within an interview, only interviewee -> interviewer. They're
  // rebuilt field by field rather than forwarded as received.
  async screen(user, msg) {
    const peer = users.get(user.peer);
    if (!peer) throw new Error('screen sharing needs an active interview');
    if (msg.role === 'sharer') requireRole(user, 'interviewee');
    else if (msg.role === 'viewer') requireRole(user, 'interviewer');
    else throw new Error('bad screen message');
    const out = { type: 'screen', from: user.id, role: msg.role };
    if (typeof msg.sdp === 'string' && msg.sdp.length < 64 * 1024) {
      out.sdp = msg.sdp;
    } else if (msg.candidate && typeof msg.candidate === 'object' &&
               typeof msg.candidate.candidate === 'string' && msg.candidate.candidate.length < 1024) {
      const c = msg.candidate;
      out.candidate = {
        candidate: c.candidate,
        sdpMid: typeof c.sdpMid === 'string' ? c.sdpMid : null,
        sdpMLineIndex: Number.isInteger(c.sdpMLineIndex) ? c.sdpMLineIndex : null,
      };
      if (typeof c.usernameFragment === 'string') out.candidate.usernameFragment = c.usernameFragment;
    } else if (msg.stop === true) {
      out.stop = true;
    } else {
      throw new Error('bad screen message');
    }
    send(peer, out);
  },
};

wss.on('connection', (ws) => {
  const user = {
    id: newId(), ws, role: null, mediaReady: false, peer: null, monitor: false, voice: null,
    calibrating: false, calibrationTimer: null,
    voiceOptions: null,                  // ids offered after calibrating
    voiceOption: null,                   // the one their voice is set to, if any
    pending: Promise.resolve(),          // handle one message at a time, in order
  };

  // the first message must say who this is
  const helloTimer = setTimeout(() => ws.close(4000, 'no hello'), 10000);
  function hello(msg) {
    clearTimeout(helloTimer);
    if (msg?.type !== 'hello' || !['interviewee', 'interviewer'].includes(msg.role)) {
      ws.close(4000, 'expected hello');
      return;
    }
    if (msg.role === 'interviewer' && !keyMatches(msg.key)) {
      send(user, { type: 'error', message: 'wrong access key' });
      ws.close(4001, 'wrong access key');
      return;
    }
    user.role = msg.role;
    if (user.role === 'interviewee') user.voice = { ...DEFAULT_VOICE };
    users.set(user.id, user);
    console.log(`[signal] ${user.role} ${user.id} connected`);
    send(user, {
      type: 'welcome', id: user.id, role: user.role, iceServers: ICE_SERVERS,
      ...(user.role === 'interviewee' ? {
        voice: user.voice, voiceSpec: VOICE_SPEC, disguise: DISGUISE,
        calibration: { phrase: CALIBRATION_PHRASE, maxSeconds: CALIBRATION_SECONDS },
      } : {}),
    });
    if (user.role === 'interviewer') send(user, { type: 'queue', length: queue.length });
  }

  ws.on('message', (data) => {
    user.pending = user.pending.then(async () => {
      let msg;
      try { msg = JSON.parse(data); } catch { return; }
      if (!user.role) return hello(msg);
      const handler = handlers[msg?.type];
      if (!handler) return send(user, { type: 'error', message: `unknown message ${msg?.type}` });
      try { await handler(user, msg); }
      catch (err) { send(user, { type: 'error', message: err.message }); }
    });
  });

  ws.on('close', () => {
    clearTimeout(helloTimer);
    clearTimeout(user.calibrationTimer);
    user.pending = user.pending.then(async () => {
      if (!user.role) return;
      user.calibrating = false;          // the media session's removal stops it
      users.delete(user.id);
      user.monitor = false;
      if (removeFromQueue(user)) broadcastQueue();
      // an interviewer dropping out isn't the interviewee's fault: put them
      // back at the front of the queue
      const requeue = user.role === 'interviewer' && user.peer;
      const peer = requeue
        ? await endInterview(user, 'the interviewer disconnected', { requeued: true })
        : await endInterview(user, 'disconnected');
      if (requeue && peer && users.has(peer.id)) {
        queue.unshift(peer.id);
        broadcastQueue();
      }
      if (user.mediaReady) await media('DELETE', `/sessions/${user.id}`).catch(() => {});
      console.log(`[signal] ${user.role} ${user.id} disconnected`);
    });
  });
  ws.on('error', () => {});
});

server.listen(PORT, () => {
  console.log(`listening on http://localhost:${PORT}, media service at ${MEDIA_URL}`);
  console.log(`interviewers: http://localhost:${PORT}/interviewer`);
  if (!process.env.INTERVIEWER_KEY) console.log(`interviewer access key (set INTERVIEWER_KEY to choose one): ${INTERVIEWER_KEY}`);
});
