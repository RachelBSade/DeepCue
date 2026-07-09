/**
 * audio_handler.js
 *
 * Captures microphone audio via Web Audio API, buffers it into fixed-length
 * windows, encodes each window as a 16-bit mono WAV, base64-encodes it, and
 * emits it via the onChunk callback.
 *
 * Target format expected by the backend Whisper pipeline:
 *   - Sample rate : 16 000 Hz
 *   - Channels    : 1 (mono)
 *   - Bit depth   : 16-bit PCM
 *   - Chunk length: CHUNK_SECONDS (default 3 s)
 */

const SAMPLE_RATE    = 16000;
const CHUNK_SECONDS  = 3;
const SAMPLES_NEEDED = SAMPLE_RATE * CHUNK_SECONDS; // 48 000 samples per chunk
const BUFFER_SIZE    = 4096; // ScriptProcessorNode buffer size

export class AudioHandler {
  /**
   * @param {function} onChunk - Called with (base64wav, chunkIndex, timestamp, sampleRate).
   */
  constructor(onChunk) {
    this._onChunk    = onChunk;
    this._chunkIndex = 0;
    this._active     = false;

    this._audioCtx   = null;
    this._stream     = null;
    this._source     = null;
    this._processor  = null;
    this._buffer     = [];
    this._bufferLen  = 0;
  }

  /** Request mic permission and begin capturing audio. */
  async start() {
    this._stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate:   SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
      video: false,
    });

    // AudioContext may not honour sampleRate on all browsers; we resample below.
    this._audioCtx  = new AudioContext({ sampleRate: SAMPLE_RATE });
    this._source    = this._audioCtx.createMediaStreamSource(this._stream);

    // ScriptProcessorNode is deprecated but universally supported.
    // AudioWorklet would require a separate .js file served over HTTP — not
    // practical for a file:// dev workflow. Replace in production if needed.
    this._processor = this._audioCtx.createScriptProcessor(BUFFER_SIZE, 1, 1);
    this._processor.onaudioprocess = (e) => this._onAudioProcess(e);

    this._source.connect(this._processor);
    this._processor.connect(this._audioCtx.destination);
    this._active = true;
  }

  /** Stop audio capture and release the microphone. */
  stop() {
    this._active = false;

    if (this._processor) { this._processor.disconnect(); this._processor = null; }
    if (this._source)    { this._source.disconnect();    this._source    = null; }
    if (this._audioCtx)  { this._audioCtx.close();       this._audioCtx  = null; }
    if (this._stream) {
      this._stream.getTracks().forEach((t) => t.stop());
      this._stream = null;
    }

    this._buffer    = [];
    this._bufferLen = 0;
    this._chunkIndex = 0;
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  /** Accumulate PCM samples; emit a WAV chunk every CHUNK_SECONDS. */
  _onAudioProcess(event) {
    if (!this._active) return;

    const inputData = event.inputBuffer.getChannelData(0); // Float32Array
    this._buffer.push(new Float32Array(inputData));
    this._bufferLen += inputData.length;

    if (this._bufferLen >= SAMPLES_NEEDED) {
      this._flushChunk();
    }
  }

  _flushChunk() {
    const pcm = _mergePCM(this._buffer, this._bufferLen);

    // Trim to exactly SAMPLES_NEEDED; discard any overflow.
    const trimmed = pcm.subarray(0, SAMPLES_NEEDED);

    const wavBuffer = _encodeWav(trimmed, SAMPLE_RATE);
    const base64    = _bufferToBase64(wavBuffer);

    this._onChunk(base64, this._chunkIndex++, Date.now() / 1000, SAMPLE_RATE);

    this._buffer    = [];
    this._bufferLen = 0;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Merge an array of Float32Arrays into a single contiguous Float32Array. */
function _mergePCM(chunks, totalLen) {
  const merged = new Float32Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

/**
 * Encode a Float32 mono PCM array to a 16-bit WAV ArrayBuffer.
 * WAV structure: RIFF header → fmt chunk → data chunk.
 */
function _encodeWav(samples, sampleRate) {
  const numSamples  = samples.length;
  const byteRate    = sampleRate * 2; // 16-bit mono → 2 bytes per sample
  const dataSize    = numSamples * 2;
  const buffer      = new ArrayBuffer(44 + dataSize);
  const view        = new DataView(buffer);

  // RIFF chunk
  _writeString(view, 0, 'RIFF');
  view.setUint32(4,  36 + dataSize, true);
  _writeString(view, 8, 'WAVE');

  // fmt sub-chunk
  _writeString(view, 12, 'fmt ');
  view.setUint32(16, 16,         true); // sub-chunk size
  view.setUint16(20, 1,          true); // PCM format
  view.setUint16(22, 1,          true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate,   true);
  view.setUint16(32, 2,          true); // block align (1 channel × 2 bytes)
  view.setUint16(34, 16,         true); // bits per sample

  // data sub-chunk
  _writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Convert Float32 [-1, 1] → Int16 and write samples
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return buffer;
}

/** Write an ASCII string into a DataView at the given byte offset (used for WAV chunk IDs). */
function _writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

/** Convert an ArrayBuffer to a base64 string. */
function _bufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary  = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}
