/**
 * main.js
 *
 * Entry point — orchestrates all DeepCue frontend modules.
 *
 * Session state machine:
 *   idle → connecting → active → ended → idle
 *
 * Session ID is a UUID4 generated client-side before the WebSocket
 * connection is opened. It is embedded in the WebSocket URL and sent
 * again in every outbound message for traceability.
 */

import { MediaPipeHandler } from './mediapipe_handler.js';
import { AudioHandler }     from './audio_handler.js';
import { WebSocketClient }  from './websocket_client.js';
import { UIController }     from './ui_controller.js';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const WS_HOST = 'localhost:8000';

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

let sessionId       = null;
let ui              = null;
let wsClient        = null;
let mediaPipe       = null;
let audioHandler    = null;

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  ui = new UIController();
  ui.onStart = handleStart;
  ui.onStop  = handleStop;
});

// ---------------------------------------------------------------------------
// Session lifecycle handlers
// ---------------------------------------------------------------------------

async function handleStart() {
  const candidateName = ui.getCandidateName() || 'Unknown';

  // Generate a fresh session ID for each interview run.
  sessionId = crypto.randomUUID();
  const wsUrl = `ws://${WS_HOST}/ws/interview/${sessionId}/`;

  ui.setConnecting();

  // --- WebSocket -----------------------------------------------------------
  wsClient = new WebSocketClient(wsUrl);

  wsClient.on('__open', () => {
    // Connection established — send session_start immediately.
    wsClient.sendSessionStart(candidateName);
  });

  wsClient.on('session_started', (_data) => {
    // Server confirmed session creation — start camera + mic.
    ui.setActive(candidateName);
    _startCapture(candidateName);
  });

  wsClient.on('emotion_result', (data) => {
    ui.updateEmotions(data.scores, data.dominant_emotion);
  });

  wsClient.on('transcript_update', (data) => {
    ui.appendTranscript(data.text, data.segment_index);
  });

  wsClient.on('session_ended', (data) => {
    ui.setEnded(data.report_url);
    _stopCapture();
  });

  wsClient.on('error', (data) => {
    ui.showError(data.message);
    console.error('[Server error]', data.message);
  });

  wsClient.on('__close', () => {
    // Only show error if session was still active (not an intentional close).
    if (mediaPipe || audioHandler) {
      ui.showError('Connection lost. Attempting to reconnect…');
    }
  });

  wsClient.on('__max_retries', () => {
    ui.showError('Could not reconnect to server. Please refresh and try again.');
    ui.setIdle();
    _stopCapture();
  });

  wsClient.connect();
}

async function handleStop() {
  if (wsClient) {
    wsClient.sendSessionEnd(sessionId);
    // wsClient remains open; server pushes session_ended when report is ready.
    // We close the WS only after receiving session_ended (handled above).
  }
  _stopCapture();
}

// ---------------------------------------------------------------------------
// Capture helpers
// ---------------------------------------------------------------------------

async function _startCapture(candidateName) {
  const videoEl = document.getElementById('webcam');

  // --- MediaPipe -----------------------------------------------------------
  mediaPipe = new MediaPipeHandler(videoEl, (landmarks, frameIndex, timestamp) => {
    if (!wsClient) return;
    wsClient.sendVideoFrame(sessionId, landmarks, frameIndex, timestamp);
  });

  // --- Audio ---------------------------------------------------------------
  audioHandler = new AudioHandler((base64wav, chunkIndex, timestamp, sampleRate) => {
    if (!wsClient) return;
    wsClient.sendAudioChunk(sessionId, base64wav, chunkIndex, timestamp, sampleRate);
  });

  try {
    // Start both in parallel; MediaPipe opening the camera grants video access,
    // AudioHandler requests mic separately.
    await Promise.all([
      mediaPipe.start(),
      audioHandler.start(),
    ]);
  } catch (err) {
    ui.showError(`Media access error: ${err.message}`);
    console.error('[Capture]', err);
    _stopCapture();
  }
}

function _stopCapture() {
  if (mediaPipe) {
    mediaPipe.stop();
    mediaPipe = null;
  }
  if (audioHandler) {
    audioHandler.stop();
    audioHandler = null;
  }
}
