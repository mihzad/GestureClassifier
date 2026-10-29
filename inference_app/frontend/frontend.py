from __future__ import annotations

import html
import os

import gradio as gr

DEFAULT_API_URL = os.getenv("PREDICT_URL", "/predict")


HTML_TEMPLATE = r"""
<div id="dactyl-live-app">
  <style>
    #dactyl-live-app {
      --bg: #0b1020;
      --panel: #121a2b;
      --panel-2: #172238;
      --line: rgba(174, 190, 220, .16);
      --muted: #8d9ab4;
      --text: #edf3ff;
      --mint: #8cf0cb;
      --mint-ink: #09251d;
      --orange: #ffbc7a;
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      width: 100%;
      box-sizing: border-box;
    }
    #dactyl-live-app * { box-sizing: border-box; }
    #dactyl-live-app .app-shell { display: flex; flex-direction: column; gap: 18px; }
    #dactyl-live-app .topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    #dactyl-live-app .brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
    #dactyl-live-app .brand-mark { width: 42px; height: 42px; border-radius: 14px; display: grid; place-items: center; background: var(--mint); color: var(--mint-ink); font-weight: 900; font-size: 21px; letter-spacing: -1px; flex: 0 0 auto; }
    #dactyl-live-app .eyebrow { color: var(--mint); font-size: 10px; font-weight: 800; letter-spacing: .19em; text-transform: uppercase; margin-bottom: 4px; }
    #dactyl-live-app h1 { padding: 0; margin: 0; font-size: clamp(21px, 2.4vw, 30px); line-height: 1.08; font-weight: 750; letter-spacing: -.04em; }
    #dactyl-live-app .topbar-note { color: var(--muted); font-size: 12px; text-align: right; line-height: 1.55; }
    #dactyl-live-app .workspace { display: grid; grid-template-columns: minmax(0, 1.75fr) minmax(280px, .82fr); gap: 16px; align-items: stretch; }
    #dactyl-live-app .panel { min-width: 0; border: 1px solid var(--line); border-radius: 20px; background: linear-gradient(160deg, rgba(25, 37, 60, .94), rgba(14, 21, 36, .98)); overflow: hidden; box-shadow: 0 18px 48px rgba(0,0,0,.13); }
    #dactyl-live-app .theatre-panel { padding: 12px; }
    #dactyl-live-app .theatre-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 5px 6px 12px; }
    #dactyl-live-app .section-title { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; font-weight: 800; }
    #dactyl-live-app .live-pill { display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--line); color: #b5bfd2; padding: 6px 10px; border-radius: 999px; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: .11em; white-space: nowrap; }
    #dactyl-live-app .live-dot { width: 7px; height: 7px; border-radius: 50%; background: #6a7892; }
    #dactyl-live-app .live-pill.is-live { color: #baf9df; border-color: rgba(140,240,203,.35); background: rgba(140,240,203,.08); }
    #dactyl-live-app .live-pill.is-live .live-dot { background: #7af0c2; box-shadow: 0 0 0 4px rgba(122,240,194,.12); }
    #dactyl-live-app .stage { position: relative; width: 100%; aspect-ratio: 16 / 9; border-radius: 14px; overflow: hidden; background: radial-gradient(ellipse at center, #18233a 0%, #080c15 72%); border: 1px solid rgba(255,255,255,.07); }
    #dactyl-live-app #dactyl-video { width: 100%; height: 100%; display: block; object-fit: contain; transform: scaleX(-1); background: #080c15; }
    #dactyl-live-app .stage-vignette { position: absolute; inset: 0; pointer-events: none; background: linear-gradient(180deg, rgba(3,7,14,.38), transparent 24%, transparent 70%, rgba(3,7,14,.48)); }
    #dactyl-live-app .stage-meta { position: absolute; left: 14px; top: 14px; display: flex; flex-direction: column; gap: 6px; pointer-events: none; }
    #dactyl-live-app .stage-tag { padding: 6px 9px; border-radius: 8px; background: rgba(5,10,20,.64); border: 1px solid rgba(255,255,255,.14); backdrop-filter: blur(8px); color: #dce7fa; font: 700 10px/1.1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .07em; text-transform: uppercase; }
    #dactyl-live-app .prediction-chip { position: absolute; top: 14px; right: 14px; max-width: 62%; min-width: 72px; text-align: center; padding: 10px 16px; border-radius: 12px; color: #08251c; background: var(--mint); font-size: clamp(16px, 2.3vw, 28px); font-weight: 850; line-height: 1.1; box-shadow: 0 8px 28px rgba(0,0,0,.24); overflow-wrap: anywhere; pointer-events: none; transition: background .15s ease, color .15s ease, transform .15s ease; }
    #dactyl-live-app .prediction-chip.is-empty { background: rgba(17,26,43,.88); color: #b0bdd2; border: 1px solid rgba(255,255,255,.15); font-size: 12px; font-weight: 750; }
    #dactyl-live-app .prediction-chip.is-hidden { display: none; }
    #dactyl-live-app .stage-bottom { position: absolute; left: 14px; right: 14px; bottom: 14px; display: flex; align-items: end; justify-content: space-between; gap: 12px; pointer-events: none; }
    #dactyl-live-app .stage-caption { color: #d5deef; font-size: 11px; text-shadow: 0 2px 8px #000; }
    #dactyl-live-app .progress-track { width: min(32%, 160px); height: 4px; border-radius: 9px; overflow: hidden; background: rgba(255,255,255,.2); }
    #dactyl-live-app .progress-fill { width: 0%; height: 100%; background: var(--mint); transition: width .05s linear; }
    #dactyl-live-app .controls { display: flex; align-items: end; flex-wrap: wrap; gap: 10px; padding: 12px 2px 2px; }
    #dactyl-live-app .control-group { display: flex; flex-direction: column; gap: 6px; min-width: 102px; }
    #dactyl-live-app .control-label { color: var(--muted); font-weight: 750; font-size: 10px; text-transform: uppercase; letter-spacing: .1em; }
    #dactyl-live-app input, #dactyl-live-app textarea { font: inherit; }
    #dactyl-live-app .interval-input { width: 112px; color: var(--text); background: #0c1322; border: 1px solid var(--line); border-radius: 10px; min-height: 40px; padding: 0 10px; outline: none; }
    #dactyl-live-app .interval-input:focus, #dactyl-live-app .api-input:focus, #dactyl-live-app .transcript:focus { border-color: rgba(140,240,203,.62); box-shadow: 0 0 0 3px rgba(140,240,203,.08); }
    #dactyl-live-app .btn { display: inline-flex; gap: 7px; justify-content: center; align-items: center; min-height: 40px; border: 1px solid transparent; border-radius: 11px; padding: 0 14px; font-size: 12px; font-weight: 800; cursor: pointer; transition: transform .12s ease, filter .12s ease, opacity .12s ease; white-space: nowrap; }
    #dactyl-live-app .btn:hover { filter: brightness(1.07); transform: translateY(-1px); }
    #dactyl-live-app .btn:disabled { opacity: .4; cursor: not-allowed; transform: none; }
    #dactyl-live-app .btn-primary { background: var(--mint); color: var(--mint-ink); }
    #dactyl-live-app .btn-secondary { background: #202d45; color: #e7efff; border-color: var(--line); }
    #dactyl-live-app .btn-quiet { background: transparent; color: #bcc8dc; border-color: var(--line); }
    #dactyl-live-app .btn-stop { background: rgba(255,188,122,.11); color: #ffca98; border-color: rgba(255,188,122,.22); }
    #dactyl-live-app .api-row { margin-top: 12px; display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: center; gap: 12px; padding: 11px 12px; border: 1px solid var(--line); border-radius: 12px; background: rgba(7,12,22,.36); }
    #dactyl-live-app .api-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; font-weight: 800; white-space: nowrap; }
    #dactyl-live-app .api-input { width: 100%; min-width: 0; border: 1px solid transparent; border-radius: 8px; background: transparent; color: #e0e9fa; outline: none; padding: 5px 8px; font: 11px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
    #dactyl-live-app .tiny-note { color: var(--muted); font-size: 10px; line-height: 1.55; padding: 8px 3px 0; }
    #dactyl-live-app .transcript-panel { display: flex; flex-direction: column; min-height: 100%; padding: 18px; }
    #dactyl-live-app .transcript-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; margin-bottom: 16px; }
    #dactyl-live-app .transcript-head h2 { margin: 0; font-size: 17px; letter-spacing: -.025em; }
    #dactyl-live-app .transcript-sub { color: var(--muted); font-size: 11px; line-height: 1.5; margin-top: 5px; }
    #dactyl-live-app .counter { font: 700 10px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; color: #b4c2d8; border: 1px solid var(--line); background: rgba(7,12,22,.32); padding: 7px 9px; border-radius: 8px; white-space: nowrap; }
    #dactyl-live-app .transcript-wrap { flex: 1; min-height: 250px; display: flex; flex-direction: column; }
    #dactyl-live-app .transcript { resize: vertical; flex: 1; width: 100%; min-height: 250px; max-height: 60vh; padding: 15px; border-radius: 13px; border: 1px solid var(--line); outline: none; color: #f0f6ff; background: rgba(5,10,19,.7); font-size: 20px; line-height: 1.65; letter-spacing: .035em; }
    #dactyl-live-app .transcript::placeholder { color: #58667f; font-size: 13px; letter-spacing: 0; }
    #dactyl-live-app .transcript-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 11px; }
    #dactyl-live-app .transcript-actions .btn { flex: 1; }
    #dactyl-live-app .transcript-tip { margin-top: auto; padding-top: 20px; color: var(--muted); font-size: 11px; line-height: 1.65; }
    #dactyl-live-app .transcript-tip strong { color: #dce7fa; font-weight: 700; }
    #dactyl-live-app .footer-line { display: flex; align-items: center; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 10px; line-height: 1.5; padding: 0 3px; }
    #dactyl-live-app .status-text { color: #b8c6dc; }
    #dactyl-live-app .status-text.is-error { color: #ffbd9d; }
    #dactyl-live-app .tensor-note { color: #aebbd1; font: 10px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; text-align: right; }
    #dactyl-live-app .hidden-canvas { display: none !important; }
    @media (max-width: 920px) {
      #dactyl-live-app .workspace { grid-template-columns: minmax(0, 1fr); }
      #dactyl-live-app .transcript-panel { min-height: auto; }
      #dactyl-live-app .transcript-wrap { min-height: 180px; }
      #dactyl-live-app .transcript { min-height: 180px; max-height: 360px; }
      #dactyl-live-app .transcript-tip { padding-top: 14px; }
    }
    @media (max-width: 520px) {
      #dactyl-live-app .topbar-note { display: none; }
      #dactyl-live-app .brand-mark { width: 36px; height: 36px; border-radius: 11px; }
      #dactyl-live-app .theatre-panel { padding: 8px; }
      #dactyl-live-app .controls { align-items: stretch; }
      #dactyl-live-app .control-group { min-width: 86px; }
      #dactyl-live-app .btn { padding: 0 10px; }
      #dactyl-live-app .api-row { grid-template-columns: 1fr; gap: 4px; }
      #dactyl-live-app .footer-line { align-items: flex-start; flex-direction: column; }
      #dactyl-live-app .tensor-note { text-align: left; }
    }
  </style>

  <div class="app-shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">Д</div>
        <div>
          <div class="eyebrow">Ukrainian sign recognition</div>
          <h1>Dactyl Studio <span style="color:#8291ac;font-weight:450">/ Live</span></h1>
        </div>
      </div>
      <div class="topbar-note">Real-time camera capture<br>16-frame gesture windows</div>
    </header>

    <div class="workspace">
      <section class="panel theatre-panel" aria-label="Live camera theatre">
        <div class="theatre-head">
          <div class="section-title">Camera theatre</div>
          <div id="dactyl-live-pill" class="live-pill"><span class="live-dot"></span><span id="dactyl-live-label">Camera off</span></div>
        </div>
        <div class="stage">
          <video id="dactyl-video" autoplay playsinline muted></video>
          <div class="stage-vignette"></div>
          <div class="stage-meta">
            <span class="stage-tag" id="dactyl-record-tag">STANDBY</span>
            <span class="stage-tag" id="dactyl-clip-tag">CLIP 00</span>
          </div>
          <div id="dactyl-prediction" class="prediction-chip is-hidden" aria-live="polite">—</div>
          <div class="stage-bottom">
            <div class="stage-caption" id="dactyl-stage-caption">Start the camera to begin</div>
            <div class="progress-track" aria-label="Current clip progress"><div class="progress-fill" id="dactyl-progress"></div></div>
          </div>
        </div>

        <div class="controls">
          <div class="control-group">
            <label class="control-label" for="dactyl-interval">Clip length · sec</label>
            <input id="dactyl-interval" class="interval-input" type="number" value="1" min="0.5" max="10" step="0.25" inputmode="decimal">
          </div>
          <button id="dactyl-start" class="btn btn-primary" type="button">▶ Start camera</button>
          <button id="dactyl-stop" class="btn btn-stop" type="button" disabled>■ Stop</button>
          <button id="dactyl-space" class="btn btn-secondary" type="button" title="Insert a space in the transcript">+ Space</button>
        </div>

        <div class="api-row">
          <label class="api-label" for="dactyl-api-url">Prediction endpoint</label>
          <input id="dactyl-api-url" class="api-input" type="text" value="__PREDICT_URL__" spellcheck="false" aria-label="FastAPI prediction endpoint">
        </div>
        <div class="tiny-note">Each clip is sampled uniformly into 16 RGB frames. The browser resizes frames to a maximum side of 320 px, converts them to a NumPy-compatible float32 tensor, and sends a <code>video.npy</code> upload.</div>
        <canvas id="dactyl-capture-canvas" class="hidden-canvas"></canvas>
      </section>

      <aside class="panel transcript-panel" aria-label="Gesture transcript">
        <div class="transcript-head">
          <div>
            <div class="eyebrow">Output</div>
            <h2>Gesture transcript</h2>
            <div class="transcript-sub">Predicted letters are appended here. Edit the text at any time.</div>
          </div>
          <div class="counter"><span id="dactyl-letter-count">0</span> chars</div>
        </div>
        <div class="transcript-wrap">
          <textarea id="dactyl-transcript" class="transcript" placeholder="Your recognized Ukrainian dactyl letters will appear here…" spellcheck="false" aria-label="Editable gesture transcript"></textarea>
        </div>
        <div class="transcript-actions">
          <button id="dactyl-copy" class="btn btn-secondary" type="button">Copy text</button>
          <button id="dactyl-download" class="btn btn-quiet" type="button">Export .txt</button>
          <button id="dactyl-clear" class="btn btn-quiet" type="button">Clear</button>
        </div>
        <div class="transcript-tip"><strong>Reading the overlay</strong><br>A recognized letter appears in the upper-right corner of the video. A model response of <code>_</code> is treated as “No gesture” and is not added to the transcript.</div>
      </aside>
    </div>

    <div class="footer-line">
      <div id="dactyl-status" class="status-text" role="status" aria-live="polite">Ready. Camera access starts only when you press “Start camera”.</div>
      <div class="tensor-note">gesture recognition studio prototype created by mihzad</div>
    </div>
  </div>
</div>
"""


JS_ON_LOAD = r"""
(() => {
  const root = element.querySelector('#dactyl-live-app');
  if (!root || root.dataset.initialized === 'true') return;
  root.dataset.initialized = 'true';

  const $ = (selector) => root.querySelector(selector);
  const video = $('#dactyl-video');
  const canvas = $('#dactyl-capture-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const startBtn = $('#dactyl-start');
  const stopBtn = $('#dactyl-stop');
  const intervalInput = $('#dactyl-interval');
  const apiInput = $('#dactyl-api-url');
  const transcript = $('#dactyl-transcript');
  const letterCount = $('#dactyl-letter-count');
  const predictionChip = $('#dactyl-prediction');
  const livePill = $('#dactyl-live-pill');
  const liveLabel = $('#dactyl-live-label');
  const recordTag = $('#dactyl-record-tag');
  const clipTag = $('#dactyl-clip-tag');
  const caption = $('#dactyl-stage-caption');
  const progress = $('#dactyl-progress');
  const status = $('#dactyl-status');

  let stream = null;
  let running = false;
  let queueWorkerActive = false;
  let clipNumber = 0;
  let predictionNumber = 0;
  let workQueue = [];
  let cameraLoopPromise = null;
  const CAPTURE_EVERY_MS = 50;
  const MAX_QUEUED_CLIPS = 2;
  const MAX_SIDE = 320;
  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  function setStatus(message, isError = false) {
    status.textContent = message;
    status.classList.toggle('is-error', !!isError);
  }

  function updateCount() {
    letterCount.textContent = String(Array.from(transcript.value).length);
  }

  function setLive(isLive) {
    livePill.classList.toggle('is-live', isLive);
    liveLabel.textContent = isLive ? 'Recording' : 'Camera off';
    recordTag.textContent = isLive ? '● RECORDING' : 'STANDBY';
    startBtn.disabled = isLive;
    stopBtn.disabled = !isLive;
    intervalInput.disabled = !isLive ? false : false;
    caption.textContent = isLive ? 'Capturing live gesture windows' : 'Start the camera to begin';
  }

  function hidePrediction() {
    predictionChip.classList.add('is-hidden');
    predictionChip.classList.remove('is-empty');
    predictionChip.textContent = '';
  }

  function showPrediction(value) {
    predictionChip.classList.remove('is-hidden');
    if (value === '_') {
      predictionChip.classList.add('is-empty');
      predictionChip.textContent = 'No gesture';
    } else {
      predictionChip.classList.remove('is-empty');
      predictionChip.textContent = value;
    }
  }

  function appendPrediction(value) {
    if (!value || value.trim() === '' || value === '_') return;
    // Predictions are letters, so concatenate them directly rather than adding spaces.
    transcript.value += value;
    transcript.focus({ preventScroll: true });
    transcript.setSelectionRange(transcript.value.length, transcript.value.length);
    updateCount();
  }

  function captureFrame() {
    if (!video.videoWidth || !video.videoHeight || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return null;
    const scale = Math.min(1, MAX_SIDE / Math.max(video.videoWidth, video.videoHeight));
    const width = Math.max(1, Math.round(video.videoWidth * scale));
    const height = Math.max(1, Math.round(video.videoHeight * scale));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.drawImage(video, 0, 0, width, height);
    const image = ctx.getImageData(0, 0, width, height);
    return { width, height, pixels: new Uint8ClampedArray(image.data) };
  }

  function selectUniformly(frames, count = 16) {
    if (!frames.length) return [];
    if (frames.length === 1) return Array.from({ length: count }, () => frames[0]);
    return Array.from({ length: count }, (_, i) => {
      const index = Math.round(i * (frames.length - 1) / (count - 1));
      return frames[index];
    });
  }

  function framesToNpy(selectedFrames) {
    const T = 16;
    const H = selectedFrames[0].height;
    const W = selectedFrames[0].width;
    // C-order tensor with shape (C=3, T=16, H, W). No batch dimension.
    // Each channel plane contains all 16 frames, matching numpy array [C,T,H,W].
    const floatData = new Float32Array(3 * T * H * W);
    let out = 0;
    for (let c = 0; c < 3; c++) {
      for (let t = 0; t < T; t++) {
        const frame = selectedFrames[t];
        for (let y = 0; y < H; y++) {
          let pixelOffset = (y * W) * 4 + c;
          for (let x = 0; x < W; x++) {
            floatData[out++] = frame.pixels[pixelOffset] / 255.0;
            pixelOffset += 4;
          }
        }
      }
    }

    const shape = `(3, ${T}, ${H}, ${W})`;
    const headerBase = `{'descr': '<f4', 'fortran_order': False, 'shape': ${shape}, }`;
    const encoder = new TextEncoder();
    let header = headerBase;
    let headerBytes = encoder.encode(header);
    const padding = (16 - ((10 + headerBytes.length + 1) % 16)) % 16;
    header += ' '.repeat(padding) + '\n';
    headerBytes = encoder.encode(header);

    // NumPy .npy format v1.0: magic(6), version(2), uint16 header length, header, data.
    const payload = new Uint8Array(floatData.buffer);
    const npy = new Uint8Array(10 + headerBytes.length + payload.byteLength);
    npy.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
    new DataView(npy.buffer).setUint16(8, headerBytes.length, true);
    npy.set(headerBytes, 10);
    npy.set(payload, 10 + headerBytes.length);
    return npy;
  }

  async function sendClip(frames, clipId) {
    const selected = selectUniformly(frames, 16);
    if (selected.length !== 16) return;
    const npyBytes = framesToNpy(selected);
    const apiUrl = apiInput.value.trim() || '/predict';
    const form = new FormData();
    form.append('vid', new Blob([npyBytes], { type: 'application/octet-stream' }), 'video.npy');

    const startedAt = performance.now();
    const response = await fetch(apiUrl, { method: 'POST', body: form });
    if (!response.ok) {
      const detail = await response.text().catch(() => '');
      throw new Error(`API returned ${response.status}${detail ? ': ' + detail.slice(0, 180) : ''}`);
    }
    const body = await response.json();
    if (!body || !Array.isArray(body.predictions) || body.predictions.length === 0) {
      throw new Error('API response must contain a non-empty predictions array.');
    }
    const prediction = String(body.predictions[0]);
    const elapsed = Math.round(performance.now() - startedAt);
    predictionNumber += 1;
    showPrediction(prediction);
    appendPrediction(prediction);
    setStatus(prediction === '_'
      ? `Clip ${clipId} · no gesture · ${elapsed} ms`
      : `Clip ${clipId} · prediction “${prediction}” · ${elapsed} ms`);
  }

  async function runQueue() {
    if (queueWorkerActive) return;
    queueWorkerActive = true;
    try {
      while (workQueue.length) {
        const item = workQueue.shift();
        try {
          await sendClip(item.frames, item.clipId);
        } catch (error) {
          setStatus(`Clip ${item.clipId} failed: ${error && error.message ? error.message : error}`, true);
          showPrediction('API error');
        }
      }
    } finally {
      queueWorkerActive = false;
      if (workQueue.length) runQueue();
    }
  }

  function enqueueClip(frames) {
    clipNumber += 1;
    clipTag.textContent = `CLIP ${String(clipNumber).padStart(2, '0')}`;
    if (!frames.length) {
      setStatus(`Clip ${clipNumber} contained no camera frames.`, true);
      return;
    }
    if (workQueue.length >= MAX_QUEUED_CLIPS) {
      setStatus(`Clip ${clipNumber} skipped because the model queue is full. Increase clip length or speed up inference.`, true);
      return;
    }
    workQueue.push({ frames, clipId: clipNumber });
    setStatus(`Clip ${clipNumber} captured · ${frames.length} source frames · waiting for prediction`);
    runQueue();
  }

  async function captureLoop() {
    while (running) {
      const seconds = Math.min(10, Math.max(0.5, Number(intervalInput.value) || 1));
      intervalInput.value = String(seconds);
      const duration = seconds * 1000;
      const began = performance.now();
      const frames = [];
      while (running && performance.now() - began < duration) {
        const frame = captureFrame();
        if (frame) frames.push(frame);
        const elapsed = performance.now() - began;
        progress.style.width = `${Math.min(100, 100 * elapsed / duration)}%`;
        await sleep(Math.min(CAPTURE_EVERY_MS, Math.max(5, duration - elapsed)));
      }
      progress.style.width = '0%';
      if (running) enqueueClip(frames);
    }
  }

  async function startCamera() {
    if (running) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('Camera access requires HTTPS or localhost in a browser that supports getUserMedia.', true);
      return;
    }
    startBtn.disabled = true;
    setStatus('Requesting camera permission…');
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, max: 30 }
        }
      });
      video.srcObject = stream;
      await video.play();
      running = true;
      setLive(true);
      hidePrediction();
      setStatus('Camera is live. Capturing the first clip…');
      cameraLoopPromise = captureLoop();
    } catch (error) {
      startBtn.disabled = false;
      setLive(false);
      const message = error && error.message ? error.message : String(error);
      setStatus(`Could not start camera: ${message}`, true);
    }
  }

  function stopCamera() {
    running = false;
    workQueue = [];
    if (stream) {
      for (const track of stream.getTracks()) track.stop();
      stream = null;
    }
    video.pause();
    video.srcObject = null;
    progress.style.width = '0%';
    setLive(false);
    setStatus('Camera stopped. Transcript is preserved.');
  }

  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);
  intervalInput.addEventListener('change', () => {
    const seconds = Math.min(10, Math.max(0.5, Number(intervalInput.value) || 1));
    intervalInput.value = String(seconds);
  });
  transcript.addEventListener('input', updateCount);
  $('#dactyl-space').addEventListener('click', () => {
    const a = transcript.selectionStart;
    const b = transcript.selectionEnd;
    transcript.setRangeText(' ', a, b, 'end');
    transcript.focus();
    updateCount();
  });
  $('#dactyl-clear').addEventListener('click', () => {
    transcript.value = '';
    updateCount();
    transcript.focus();
  });
  $('#dactyl-copy').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(transcript.value);
      setStatus('Transcript copied to clipboard.');
    } catch (error) {
      transcript.focus();
      transcript.select();
      setStatus('Clipboard permission was blocked; transcript selected so you can copy it.', true);
    }
  });
  $('#dactyl-download').addEventListener('click', () => {
    const blob = new Blob([transcript.value], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'dactyl-transcript.txt';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    setStatus('Transcript exported as dactyl-transcript.txt.');
  });
  window.addEventListener('beforeunload', () => {
    if (stream) for (const track of stream.getTracks()) track.stop();
  }, { once: true });

  updateCount();
  setLive(false);
})();
"""


GLOBAL_CSS = r"""
.gradio-container { max-width: 1680px !important; width: 100% !important; padding: 18px clamp(10px, 2vw, 28px) !important; }
footer { display: none !important; }
body, .gradio-container { background: #0b1020 !important; }
"""


def build_demo(default_api_url: str | None = None) -> gr.Blocks:
    """Build the Gradio frontend. Mount it into the FastAPI app for same-origin API calls."""
    api_url = html.escape(default_api_url if default_api_url is not None else DEFAULT_API_URL, quote=True)
    markup = HTML_TEMPLATE.replace("__PREDICT_URL__", api_url)

    with gr.Blocks(
        title="Dactyl Studio · Live",
        fill_width=True,
        fill_height=True,
    ) as demo:
        gr.HTML(
            value=markup,
            js_on_load=JS_ON_LOAD,
            apply_default_css=False,
            container=False,
            elem_id="dactyl-live-component",
        )
    return demo


demo = build_demo(default_api_url='http://127.0.0.1:8000/predict')


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        css=GLOBAL_CSS,
    )