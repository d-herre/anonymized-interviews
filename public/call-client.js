// call-client.js -- browser side of the interview site, used by both pages.
//
// The browser keeps one WebRTC connection, to the server's media service
// (not to the other person). It sends the raw microphone up that connection
// and plays the one audio track that comes back down: the other person's
// voice during an interview (the interviewee's disguised by rt_lpc), the
// interviewee's own disguised voice while they try it out, silence otherwise.
// Starting or ending an interview only changes routing on the server, so the
// WebRTC connection stays up.
//
//   interviewee:
//     const client = new CallClient(`wss://${location.host}/signal`, { role: 'interviewee' });
//     await client.connect();
//     await client.startAudio();          // from a click
//     client.setVoice({ formant: 1.2 });  // adjust the disguise (before joining or while waiting)
//     client.monitor(true);               //   ...and listen to it
//     client.join();                      // enter the queue when ready
//     client.calibrate('start');          // measure their voice while they read
//     client.calibrate('stop');           //   client.calibration.phrase -> 'calibration' event
//     client.chooseVoice('bright');       //   with the voice options; switch between them
//     await client.startScreenShare();    // during the interview, from a click
//   interviewer:
//     const client = new CallClient(url, { role: 'interviewer', key });
//     await client.connect();             // rejects if the key is wrong
//     await client.startAudio();
//     client.next();                      // take the next interviewee
//   both:
//     audioElement.srcObject = client.remoteStream;   // what this person hears
//     await client.setAudioInput({ deviceId, noiseSuppression: false });
//                                         // switch microphone or its processing, any time
//     await CallClient.audioInputs();     // the microphones to choose from
//     client.hangup();
//
// Screen sharing (interviewee -> interviewer only) goes DIRECTLY between the
// two browsers on a separate WebRTC connection; the server only relays its
// setup messages and never sees the video. Video only: no screen audio, so
// nothing can carry sound around rt_lpc.
//
// Events (CustomEvent, data in e.detail):
//   'queue' {position, length}   interviewees: place in the queue (null = not in it)
//                                interviewers: {length}
//   'connected' {peer}  'ended' {peer, reason, requeued}
//   'voice' {voice, option}   the disguise as applied by the server, and the
//                voice option it is (null if pitch or formant were set by hand)
//   'calibration' {state, ...}   (interviewee) voice measurement: 'recording'
//                {maxSeconds}, 'analyzing', 'done' {measured, voices,
//                suggested}, 'failed' {message}, 'cancelled'.
//                voices: [{id, name, description, settings, adjusted, tags}];
//                the suggested one is applied straight away
//   'voiceOption' {id, settings, adjusted}   a voice option was applied, with
//                the settings worked out for it
//   'recalibrate' {message}   the measurement is gone: calibrate again
//   'screen' {stream, from}      (interviewer) the interviewee started sharing
//   'screen-ended' {from}        ...and stopped
//   'screen-sharing' {sharing, stream}   (interviewee) own share started/stopped
//   'microphone' {deviceId, label, requested, applied, stream}
//                                the microphone in use changed (or started):
//                                requested = audioOptions asked for, applied =
//                                what the browser actually did (getSettings())
//   'error' {message}   'state' {connection}  (WebRTC connection state)
//   'closed' {reason}   signaling connection lost

export class CallClient extends EventTarget {
  constructor(signalUrl, { role = 'interviewee', key } = {}) {
    super();
    this.signalUrl = signalUrl;
    this.role = role;
    this.key = key;
    this.id = null;
    this.ws = null;
    this.pc = null;
    this.micStream = null;
    this.micSender = null;    // the RTCRtpSender carrying the microphone
    this.muted = false;
    // what to ask the browser for. Echo cancellation keeps the other
    // person's voice, playing on our speakers, out of our microphone;
    // noise suppression and automatic gain control are the browser's own
    // processing. Change with setAudioInput().
    this.audioOptions = { deviceId: '', echoCancellation: true, noiseSuppression: true, autoGainControl: true };
    this.remoteStream = new MediaStream();   // attach to an <audio> element any time
    this.iceServers = [];
    this.pendingAnswer = null;
    this.voice = {};          // current voice settings (confirmed by the server)
    this.voiceSpec = {};      // allowed settings and ranges, from the server
    this.disguise = null;     // how far formant or pitch must be from 1 (factors)
    this.calibration = null;  // { phrase, maxSeconds } for voice measurement
    this.voiceOption = null;  // id of the voice option in use, if any
    this.pendingVoice = null; // changes not sent yet
    this.voiceTimer = null;
    this.peer = null;         // the other person, during an interview
    this.screenOut = null;    // { pc, stream, pending }  while this user shares
    this.screenIn = null;     // { pc, stream, pending, from }  while viewing theirs
  }

  // ---- signaling ----
  connect() {
    return new Promise((resolve, reject) => {
      const ws = (this.ws = new WebSocket(this.signalUrl));
      let welcomed = false;
      ws.onopen = () => {
        // say who this is; the server answers with 'welcome' or closes
        ws.send(JSON.stringify({ type: 'hello', role: this.role, ...(this.key ? { key: this.key } : {}) }));
      };
      ws.onerror = () => reject(new Error(`could not connect to ${this.signalUrl}`));
      ws.onclose = (e) => {
        if (!welcomed) reject(new Error(e.reason || 'the server closed the connection'));
        this.#emit('closed', { reason: e.reason });
      };
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'welcome') {
          welcomed = true;
          this.id = msg.id;
          this.role = msg.role;
          this.iceServers = msg.iceServers || [];
          this.voice = msg.voice || {};
          this.voiceSpec = msg.voiceSpec || {};
          this.disguise = msg.disguise || null;
          this.calibration = msg.calibration || null;
          resolve(this.id);
        } else if (msg.type === 'voice') {
          this.voice = msg.voice;
          this.voiceOption = msg.option ?? null;
          this.#emit('voice', { voice: msg.voice, option: this.voiceOption });
        } else if (msg.type === 'screen') {
          this.#onScreenSignal(msg).catch((err) =>
            this.#emit('error', { message: `screen sharing: ${err.message}` }));
        } else if (msg.type === 'answer') {
          this.pendingAnswer?.resolve(msg.sdp);
        } else if (msg.type === 'error') {
          if (!welcomed) return;              // the close that follows rejects connect()
          this.pendingAnswer?.reject(new Error(msg.message));
          this.#emit('error', { message: msg.message });
        } else {
          // an interview ending also ends any screen sharing in it
          if (msg.type === 'connected') this.peer = msg.peer;
          if (msg.type === 'ended' && msg.peer === this.peer) {
            this.peer = null;
            this.stopScreenShare({ notify: false });
            this.#closeIncomingScreen();
          }
          const { type, ...detail } = msg;
          this.#emit(type, detail);
        }
      };
    });
  }

  #send(msg) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  #emit(type, detail = {}) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }

  // ---- media: one WebRTC connection to the server ----
  // Call from a user gesture (a click), since it asks for the microphone and
  // browsers restrict audio playback until the user has interacted.
  async startAudio() {
    if (this.pc) return;

    this.micStream = await this.#openMicrophone();
    const pc = (this.pc = new RTCPeerConnection({ iceServers: this.iceServers }));
    pc.onconnectionstatechange = () => this.#emit('state', { connection: pc.connectionState });

    // one audio transceiver: microphone up, processed audio down
    const transceiver = pc.addTransceiver(this.micStream.getAudioTracks()[0], {
      direction: 'sendrecv',
      streams: [this.micStream],
    });
    this.micSender = transceiver.sender;
    this.#emit('microphone', this.microphone());
    pc.ontrack = (e) => {
      this.remoteStream.getTracks().forEach((t) => this.remoteStream.removeTrack(t));
      this.remoteStream.addTrack(e.track);
    };

    // The media service (aiortc) doesn't take candidates one at a time, so
    // gather them all first and send one complete offer.
    await pc.setLocalDescription(await pc.createOffer());
    await waitForIceGathering(pc, 3000);

    const sdp = await new Promise((resolve, reject) => {
      this.pendingAnswer = { resolve, reject };
      this.#send({ type: 'offer', sdp: pc.localDescription.sdp });
    }).finally(() => { this.pendingAnswer = null; });
    await pc.setRemoteDescription({ type: 'answer', sdp });
  }

  // mute without tearing anything down (the server keeps receiving silence)
  setMuted(muted) {
    this.muted = !!muted;
    this.micStream?.getAudioTracks().forEach((t) => { t.enabled = !this.muted; });
  }

  // ---- microphone ----
  // The microphones the browser offers (before the first getUserMedia, most
  // browsers hide them, or their names).
  static async audioInputs() {
    try {
      return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput');
    } catch {
      return [];
    }
  }

  // Changes the microphone or its processing: any of audioOptions, e.g.
  // { deviceId: '...' } ('' = the browser's default) or
  // { echoCancellation: false }. Before startAudio() it just records the
  // choice. During a session it opens the microphone again with the new
  // options and swaps the new track into the connection (no renegotiation,
  // nothing to tell the server); there's a gap of a few milliseconds. If
  // the new options can't be opened, the old ones are restored and the
  // error is thrown. Resolves to microphone().
  setAudioInput(changes) {
    // one change at a time, in order
    const run = () => this.#changeMicrophone(changes);
    const done = (this.micChange ?? Promise.resolve()).then(run, run);
    this.micChange = done.catch(() => {});
    return done;
  }

  async #changeMicrophone(changes) {
    const previous = this.audioOptions;
    this.audioOptions = { ...previous, ...changes };
    if (!this.micStream) return this.microphone();

    // Close the old capture first: some browsers won't open the same device
    // twice with different processing.
    this.micStream.getTracks().forEach((t) => t.stop());
    let stream;
    try {
      stream = await this.#openMicrophone();
    } catch (err) {
      this.audioOptions = previous;
      stream = await this.#openMicrophone().catch(() => null);
      if (stream) await this.#useMicrophone(stream);
      throw err;
    }
    await this.#useMicrophone(stream);
    return this.microphone();
  }

  // the microphone in use: see the 'microphone' event
  microphone() {
    const track = this.micStream?.getAudioTracks()[0];
    const s = track?.getSettings?.() ?? {};
    return {
      deviceId: s.deviceId ?? this.audioOptions.deviceId,
      label: track?.label ?? '',
      requested: { ...this.audioOptions },
      applied: {
        echoCancellation: s.echoCancellation,
        noiseSuppression: s.noiseSuppression,
        autoGainControl: s.autoGainControl,
      },
      stream: this.micStream,
    };
  }

  async #openMicrophone() {
    const o = this.audioOptions;
    const audio = {
      channelCount: 1,
      echoCancellation: o.echoCancellation,
      noiseSuppression: o.noiseSuppression,
      autoGainControl: o.autoGainControl,
    };
    if (!o.deviceId) return navigator.mediaDevices.getUserMedia({ audio, video: false });
    try {
      return await navigator.mediaDevices.getUserMedia({ audio: { ...audio, deviceId: { exact: o.deviceId } }, video: false });
    } catch (err) {
      // a remembered microphone that isn't there any more: use the default
      if (err.name !== 'OverconstrainedError' && err.name !== 'NotFoundError') throw err;
      this.audioOptions = { ...o, deviceId: '' };
      return navigator.mediaDevices.getUserMedia({ audio, video: false });
    }
  }

  async #useMicrophone(stream) {
    const track = stream.getAudioTracks()[0];
    track.enabled = !this.muted;
    await this.micSender?.replaceTrack(track);
    this.micStream = stream;
    this.#emit('microphone', this.microphone());
  }

  // ---- voice settings ----
  // Changes this user's own voice: any of the settings in voiceSpec, e.g.
  // { formant: 1.2, method: 'poles' }. Takes effect within a block or two
  // (tens of milliseconds), in a call or while monitoring. Calls are
  // coalesced and sent at most every 50 ms, so it's fine to call this on
  // every slider 'input' event.
  setVoice(changes) {
    this.pendingVoice = { ...this.pendingVoice, ...changes };
    if (this.voiceTimer) return;
    const flush = () => {
      this.voiceTimer = null;
      if (!this.pendingVoice) return;
      this.#send({ type: 'voice', settings: this.pendingVoice });
      this.pendingVoice = null;
      this.voiceTimer = setTimeout(flush, 50);   // trailing send for changes made meanwhile
    };
    flush();
  }

  // ---- screen sharing (peer-to-peer) ----
  // Asks the browser for a screen, window or tab to share and sends it to the
  // other user in the current call. Must be called from a user gesture.
  // Throws if not in a call or if the user cancels the picker.
  async startScreenShare() {
    if (this.role !== 'interviewee') throw new Error('only interviewees share their screen');
    if (!this.peer) throw new Error('screen sharing needs an active interview');
    if (this.screenOut) return;
    if (!navigator.mediaDevices?.getDisplayMedia)
      throw new Error('this browser cannot share its screen');

    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { ideal: 15, max: 30 } },
      audio: false,
    });
    const track = stream.getVideoTracks()[0];
    if (!this.peer) {                       // call ended while the picker was open
      stream.getTracks().forEach((t) => t.stop());
      throw new Error('the interview ended');
    }
    // screen content: keep text sharp, let the frame rate drop instead
    if ('contentHint' in track) track.contentHint = 'detail';

    const pc = new RTCPeerConnection({ iceServers: this.iceServers });
    const share = (this.screenOut = { pc, stream, pending: [] });
    pc.addTransceiver(track, { direction: 'sendonly', streams: [stream] });
    pc.onicecandidate = (e) => {
      if (e.candidate) this.#send({ type: 'screen', role: 'sharer', candidate: e.candidate.toJSON() });
    };
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'failed' && this.screenOut === share) {
        this.#emit('error', { message: 'screen sharing: could not connect to the other user' });
        this.stopScreenShare();
      }
    };
    // the browser's own "Stop sharing" button ends the track
    track.addEventListener('ended', () => { if (this.screenOut === share) this.stopScreenShare(); });

    await pc.setLocalDescription(await pc.createOffer());
    this.#send({ type: 'screen', role: 'sharer', sdp: pc.localDescription.sdp });
    this.#emit('screen-sharing', { sharing: true, stream });
  }

  stopScreenShare({ notify = true } = {}) {
    const share = this.screenOut;
    if (!share) return;
    this.screenOut = null;
    share.stream.getTracks().forEach((t) => t.stop());
    share.pc.close();
    if (notify) this.#send({ type: 'screen', role: 'sharer', stop: true });
    this.#emit('screen-sharing', { sharing: false, stream: null });
  }

  #closeIncomingScreen() {
    const view = this.screenIn;
    if (!view) return;
    this.screenIn = null;
    view.pc.close();
    this.#emit('screen-ended', { from: view.from });
  }

  // Candidates can arrive before the description they belong to has been
  // applied, and browsers reject them then, so hold them until it has.
  async #addCandidate(conn, candidate) {
    if (conn.pc.remoteDescription) await conn.pc.addIceCandidate(candidate);
    else conn.pending.push(candidate);
  }
  async #flushCandidates(conn) {
    const queued = conn.pending;
    conn.pending = [];
    for (const c of queued) await conn.pc.addIceCandidate(c).catch(() => {});
  }

  // setup messages relayed from the other user. role is THEIR side:
  // 'sharer' messages are about the screen we view, 'viewer' messages about
  // the screen we share
  async #onScreenSignal(msg) {
    if (msg.from !== this.peer) return;

    if (msg.role === 'sharer') {
      if (msg.sdp) {
        this.#closeIncomingScreen();                 // a new share replaces an old one
        const pc = new RTCPeerConnection({ iceServers: this.iceServers });
        const view = (this.screenIn = { pc, stream: null, pending: [], from: msg.from });
        pc.ontrack = (e) => {
          if (this.screenIn !== view) return;
          view.stream = e.streams[0] || new MediaStream([e.track]);
          this.#emit('screen', { stream: view.stream, from: msg.from });
        };
        pc.onicecandidate = (e) => {
          if (e.candidate) this.#send({ type: 'screen', role: 'viewer', candidate: e.candidate.toJSON() });
        };
        pc.onconnectionstatechange = () => {
          if (pc.connectionState === 'failed' && this.screenIn === view) {
            this.#emit('error', { message: 'screen sharing: could not connect to the other user' });
            this.#closeIncomingScreen();
          }
        };
        await pc.setRemoteDescription({ type: 'offer', sdp: msg.sdp });
        await this.#flushCandidates(view);
        await pc.setLocalDescription(await pc.createAnswer());
        this.#send({ type: 'screen', role: 'viewer', sdp: pc.localDescription.sdp });
      } else if (msg.candidate && this.screenIn) {
        await this.#addCandidate(this.screenIn, msg.candidate);
      } else if (msg.stop) {
        this.#closeIncomingScreen();
      }
    } else if (msg.role === 'viewer') {
      const share = this.screenOut;
      if (!share) return;
      if (msg.sdp) {
        await share.pc.setRemoteDescription({ type: 'answer', sdp: msg.sdp });
        await this.#flushCandidates(share);
      } else if (msg.candidate) {
        await this.#addCandidate(share, msg.candidate);
      }
    }
  }

  // Would the disguise still count as one after these changes? (The server
  // has the final say; this lets a page warn before sending.)
  isDisguised(changes = {}) {
    if (!this.disguise) return true;
    const v = { ...this.voice, ...changes };
    const far = (x, f) => Math.max(x, 1 / x) >= f;
    return far(v.formant, this.disguise.formant) || far(v.pitch, this.disguise.pitch);
  }

  // ---- queue and interviews ----
  // interviewee: measure their voice while they read client.calibration.phrase.
  // action is 'start', 'stop' (-> the voice options) or 'cancel'. Results
  // arrive as 'calibration' events. Once is enough: chooseVoice() then
  // switches between the options, with settings worked out for this voice
  // each time ('voiceOption' and 'voice' events follow).
  calibrate(action) { this.#send({ type: 'calibrate', action }); }
  chooseVoice(id) {
    // send any changes still waiting first, so they don't land on top of the
    // option's settings
    clearTimeout(this.voiceTimer);
    this.voiceTimer = null;
    if (this.pendingVoice) this.#send({ type: 'voice', settings: this.pendingVoice });
    this.pendingVoice = null;
    this.#send({ type: 'voiceOption', id });
  }

  // interviewee: join the queue (when ready, after startAudio()) / leave it
  join() { this.#send({ type: 'join' }); }
  leave() { this.#send({ type: 'leave' }); }
  // interviewer: take the next interviewee and start the interview
  next() { this.#send({ type: 'next' }); }
  // either: end the interview
  hangup() { this.#send({ type: 'hangup' }); }
  // interviewee: hear your own disguised voice while waiting (use headphones)
  monitor(on) { this.#send({ type: 'monitor', on }); }

  close() {
    this.stopScreenShare({ notify: false });
    this.#closeIncomingScreen();
    this.ws?.close();
    this.pc?.close();
    this.micStream?.getTracks().forEach((t) => t.stop());
    this.pc = null;
  }
}

function waitForIceGathering(pc, timeoutMs) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      if (pc.iceGatheringState === 'complete') { clearTimeout(timer); pc.removeEventListener('icegatheringstatechange', done); resolve(); }
    };
    // a slow or unreachable STUN server shouldn't stall the call: go with
    // whatever candidates we have
    const timer = setTimeout(() => { pc.removeEventListener('icegatheringstatechange', done); resolve(); }, timeoutMs);
    pc.addEventListener('icegatheringstatechange', done);
  });
}
