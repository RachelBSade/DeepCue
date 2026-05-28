/**
 * ui_controller.js
 *
 * Owns all direct DOM manipulation for the DeepCue interview UI.
 * main.js calls these methods; nothing else touches the DOM directly.
 *
 * Responsibilities:
 *  - Start / Stop button state
 *  - Status badge
 *  - Session timer
 *  - Emotion confidence bars (8 classes)
 *  - Dominant emotion display
 *  - Hebrew transcript feed (RTL)
 *  - Error message display
 */

const EMOTION_CLASSES = [
  'neutral', 'confident', 'anxious', 'happy',
  'sad', 'angry', 'surprised', 'uncertain',
];

export class UIController {
  constructor() {
    // Controls
    this._btnStart  = document.getElementById('btn-start');
    this._btnStop   = document.getElementById('btn-stop');
    this._nameInput = document.getElementById('candidate-name-input');

    // Status
    this._statusBadge    = document.getElementById('status-badge');
    this._sessionTimer   = document.getElementById('session-timer');
    this._errorMessage   = document.getElementById('error-message');

    // Video
    this._candidateLabel     = document.getElementById('candidate-label');
    this._noCameraOverlay    = document.getElementById('no-camera-overlay');

    // Emotion
    this._dominantValue  = document.getElementById('dominant-value');
    this._emotionRows    = {};
    for (const e of EMOTION_CLASSES) {
      this._emotionRows[e] = document.querySelector(`.emotion-row[data-emotion="${e}"]`);
    }

    // Transcript
    this._transcriptFeed = document.getElementById('transcript-feed');

    // Timer internals
    this._timerInterval  = null;
    this._timerStart     = null;

    // Callbacks set by main.js
    this.onStart = null;
    this.onStop  = null;

    this._bindEvents();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /** Transition UI into the "connecting" state. */
  setConnecting() {
    this._setStatus('connecting', 'Connecting…');
    this._btnStart.disabled  = true;
    this._btnStop.disabled   = true;
    this._nameInput.disabled = true;
    this._clearError();
  }

  /** Transition UI into the "active" (live session) state. */
  setActive(candidateName) {
    this._setStatus('active', '● Live');
    this._btnStart.disabled = true;
    this._btnStop.disabled  = false;
    this._candidateLabel.textContent = candidateName || '—';
    this._noCameraOverlay.classList.add('hidden');
    this._startTimer();
    this._clearError();
  }

  /** Transition UI into the "ended" state. */
  setEnded(reportUrl = null) {
    this._setStatus('ended', 'Ended');
    this._btnStart.disabled  = false;
    this._btnStop.disabled   = true;
    this._nameInput.disabled = false;
    this._stopTimer();

    if (reportUrl) {
      this._appendTranscriptLine(`📄 Report ready: ${reportUrl}`);
    }
  }

  /** Transition UI into the "idle" state (initial / reset). */
  setIdle() {
    this._setStatus('idle', 'Idle');
    this._btnStart.disabled  = false;
    this._btnStop.disabled   = true;
    this._nameInput.disabled = false;
    this._stopTimer();
    this._candidateLabel.textContent = '—';
  }

  /** Display a non-fatal error message below the controls. */
  showError(message) {
    this._errorMessage.textContent = message;
    this._setStatus('error', 'Error');
  }

  /**
   * Update the emotion confidence bars.
   * @param {Object} scores          - { neutral: 0.4, confident: 0.3, … }
   * @param {string} dominantEmotion
   */
  updateEmotions(scores, dominantEmotion) {
    for (const emotion of EMOTION_CLASSES) {
      const row = this._emotionRows[emotion];
      if (!row) continue;

      const value = scores[emotion] ?? 0;
      const pct   = (value * 100).toFixed(1);

      row.querySelector('.bar-fill').style.width      = `${pct}%`;
      row.querySelector('.emotion-score').textContent = value.toFixed(2);
    }
    this._dominantValue.textContent = dominantEmotion || '—';
  }

  /**
   * Append a Hebrew transcript segment to the feed.
   * @param {string} text
   * @param {number} segmentIndex
   */
  appendTranscript(text, segmentIndex) {
    this._appendTranscriptLine(text);
  }

  /** Return the current value of the candidate name input. */
  getCandidateName() {
    return this._nameInput.value.trim();
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  _bindEvents() {
    this._btnStart.addEventListener('click', () => {
      if (this.onStart) this.onStart();
    });
    this._btnStop.addEventListener('click', () => {
      if (this.onStop) this.onStop();
    });
  }

  _setStatus(state, label) {
    this._statusBadge.className = `status-badge status-${state}`;
    this._statusBadge.textContent = label;
  }

  _startTimer() {
    this._timerStart = Date.now();
    this._timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - this._timerStart) / 1000);
      const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const ss = String(elapsed % 60).padStart(2, '0');
      this._sessionTimer.textContent = `${mm}:${ss}`;
    }, 1000);
  }

  _stopTimer() {
    if (this._timerInterval) {
      clearInterval(this._timerInterval);
      this._timerInterval = null;
    }
  }

  _clearError() {
    this._errorMessage.textContent = '';
  }

  _appendTranscriptLine(text) {
    const span = document.createElement('span');
    span.className   = 'transcript-segment';
    span.textContent = text;
    this._transcriptFeed.appendChild(span);
    // Auto-scroll to latest entry
    this._transcriptFeed.scrollTop = this._transcriptFeed.scrollHeight;
  }
}
