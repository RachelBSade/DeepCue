/**
 * websocket_client.js
 *
 * Manages the WebSocket connection lifecycle for DeepCue.
 *
 * Features:
 *  - Typed message sending (one method per message type)
 *  - Incoming message routing via registered handlers
 *  - Exponential backoff reconnect (max 5 retries, cap 30 s)
 *  - Explicit close() to suppress reconnect when user stops the session
 */

const MAX_RETRIES   = 5;
const BASE_DELAY_MS = 1000;
const MAX_DELAY_MS  = 30000;

export class WebSocketClient {
  /**
   * @param {string} url - Full WebSocket URL, e.g. ws://localhost:8000/ws/interview/<uuid>/
   */
  constructor(url) {
    this._url       = url;
    this._ws        = null;
    this._handlers  = {};   // type → function(data)
    this._retries   = 0;
    this._intentionalClose = false;
  }

  // ---------------------------------------------------------------------------
  // Connection management
  // ---------------------------------------------------------------------------

  /** Open the WebSocket connection. */
  connect() {
    this._intentionalClose = false;
    this._open();
  }

  /** Close intentionally (suppresses reconnect). */
  close() {
    this._intentionalClose = true;
    if (this._ws) {
      this._ws.close(1000, 'Session ended by client');
      this._ws = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Outbound — typed send methods
  // ---------------------------------------------------------------------------

  /**
   * @param {string} candidateName
   * @param {string} [candidateEmail] - Optional; report is emailed here when set.
   */
  sendSessionStart(candidateName, candidateEmail = '') {
    this._send({
      type:            'session_start',
      candidate_name:  candidateName,
      candidate_email: candidateEmail,
    });
  }

  /**
   * @param {string}   sessionId
   * @param {Array}    landmarks   - 468 {x, y, z} objects
   * @param {number}   frameIndex
   * @param {number}   timestamp   - Unix epoch seconds
   */
  sendVideoFrame(sessionId, landmarks, frameIndex, timestamp, frameJpeg) {
    this._send({
      type:        'video_frame',
      session_id:  sessionId,
      frame_index: frameIndex,
      timestamp,
      landmarks,
      frame_jpeg:  frameJpeg,  // base64 JPEG 224×224 — used by the video model
    });
  }

  /**
   * @param {string} sessionId
   * @param {string} audioData   - base64-encoded WAV
   * @param {number} chunkIndex
   * @param {number} timestamp
   * @param {number} sampleRate
   */
  sendAudioChunk(sessionId, audioData, chunkIndex, timestamp, sampleRate) {
    this._send({
      type:        'audio_chunk',
      session_id:  sessionId,
      chunk_index: chunkIndex,
      timestamp,
      audio_data:  audioData,
      sample_rate: sampleRate,
    });
  }

  /**
   * @param {string} sessionId
   */
  sendSessionEnd(sessionId) {
    this._send({ type: 'session_end', session_id: sessionId });
  }

  // ---------------------------------------------------------------------------
  // Inbound — handler registration
  // ---------------------------------------------------------------------------

  /**
   * Register a handler for an inbound message type.
   * @param {string}   type    - e.g. 'emotion_result', 'session_started'
   * @param {function} handler - Called with the full parsed message object.
   */
  on(type, handler) {
    this._handlers[type] = handler;
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  /** @private Open the underlying WebSocket and wire its on* event callbacks. */
  _open() {
    this._ws = new WebSocket(this._url);

    this._ws.onopen = () => {
      this._retries = 0;
      this._handlers['__open']?.();
    };

    this._ws.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        console.error('[WS] Failed to parse message:', event.data);
        return;
      }
      const handler = this._handlers[data.type];
      if (handler) {
        handler(data);
      } else {
        console.warn('[WS] No handler registered for type:', data.type);
      }
    };

    this._ws.onclose = (event) => {
      this._handlers['__close']?.(event);
      if (!this._intentionalClose) {
        this._scheduleReconnect();
      }
    };

    this._ws.onerror = (event) => {
      console.error('[WS] Error:', event);
      this._handlers['__error']?.(event);
      // onerror is always followed by onclose; reconnect is handled there.
    };
  }

  /**
   * @private Serialize and send a payload if the socket is open; warns and drops it otherwise.
   * @param {object} payload
   */
  _send(payload) {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(payload));
    } else {
      console.warn('[WS] send() called while not connected, dropping:', payload.type);
    }
  }

  /** @private Reconnect with exponential backoff (capped), or give up after MAX_RETRIES. */
  _scheduleReconnect() {
    if (this._retries >= MAX_RETRIES) {
      console.error('[WS] Max reconnect attempts reached.');
      this._handlers['__max_retries']?.();
      return;
    }

    const delay = Math.min(BASE_DELAY_MS * 2 ** this._retries, MAX_DELAY_MS);
    this._retries++;
    console.info(`[WS] Reconnecting in ${delay}ms (attempt ${this._retries}/${MAX_RETRIES})`);

    setTimeout(() => {
      if (!this._intentionalClose) this._open();
    }, delay);
  }
}
