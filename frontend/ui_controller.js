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

const LAYOUT_STORAGE_KEY     = 'deepcue_layout';
const LONG_PRESS_MS          = 450;  // hold time before a panel "arms" for a swap-drag
const SWAP_DRAG_THRESHOLD_PX = 60;   // horizontal drag distance (once armed) that triggers a swap
const MIN_CAMERA_SIZE        = 220;
const MAX_CAMERA_SIZE        = 640;
const MIN_QUESTION_SIZE      = 70;
const MAX_QUESTION_SIZE      = 240;

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
    this._speechRateValue = document.getElementById('speech-rate-value');
    this._emotionRows    = {};
    for (const e of EMOTION_CLASSES) {
      this._emotionRows[e] = document.querySelector(`.emotion-row[data-emotion="${e}"]`);
    }

    // Transcript
    this._transcriptFeed = document.getElementById('transcript-feed');

    // Layout controls — drag handles, no settings panel
    this._mainContent      = document.querySelector('.main-content');
    this._videoPanel        = document.querySelector('.video-panel');
    this._emotionPanelEl    = document.querySelector('.emotion-panel');
    this._panelResizer      = document.getElementById('panel-resizer');
    this._questionPanel     = document.getElementById('question-panel');
    this._questionResizer   = document.getElementById('question-resizer');

    // Timer internals
    this._timerInterval  = null;
    this._timerStart     = null;

    // Callbacks set by main.js
    this.onStart = null;
    this.onStop  = null;

    this._bindEvents();
    this._applyLayout(this._loadLayout());
    this._initPanelResizer();
    this._initQuestionResizer();
    this._initSwapDrag();
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
   * @param {number|null} [speechRateWpm] - words-per-minute stress signal, or null if unavailable.
   */
  updateEmotions(scores, dominantEmotion, speechRateWpm = null) {
    for (const emotion of EMOTION_CLASSES) {
      const row = this._emotionRows[emotion];
      if (!row) continue;

      const value = scores[emotion] ?? 0;
      const pct   = (value * 100).toFixed(1);

      row.querySelector('.bar-fill').style.width      = `${pct}%`;
      row.querySelector('.emotion-score').textContent = value.toFixed(2);
    }
    this._dominantValue.textContent = dominantEmotion || '—';

    if (this._speechRateValue) {
      this._speechRateValue.textContent =
        speechRateWpm != null ? Math.round(speechRateWpm) : '—';
    }
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

  /** @private Wire the Start/Stop buttons to the onStart/onStop callbacks set by main.js. */
  _bindEvents() {
    this._btnStart.addEventListener('click', () => {
      if (this.onStart) this.onStart();
    });
    this._btnStop.addEventListener('click', () => {
      if (this.onStop) this.onStop();
    });
  }

  /** @private Drag the camera/emotion divider left-right to resize the camera panel (VS Code-style). */
  _initPanelResizer() {
    let startX = 0;
    let startSize = 0;

    const onMove = (e) => {
      const dx = e.clientX - startX;
      const next = Math.min(MAX_CAMERA_SIZE, Math.max(MIN_CAMERA_SIZE, startSize + dx));
      document.documentElement.style.setProperty('--camera-size', `${next}px`);
    };
    const onUp = () => {
      this._panelResizer.classList.remove('dragging');
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);
      this._saveLayout({ cameraSize: this._readCssVarPx('--camera-size', 340) });
    };

    this._panelResizer.addEventListener('pointerdown', (e) => {
      startX = e.clientX;
      startSize = this._readCssVarPx('--camera-size', 340);
      this._panelResizer.classList.add('dragging');
      document.addEventListener('pointermove', onMove);
      document.addEventListener('pointerup', onUp);
    });
  }

  /** @private Drag the handle below the question panel up/down to resize its height. */
  _initQuestionResizer() {
    let startY = 0;
    let startSize = 0;

    const onMove = (e) => {
      const dy = e.clientY - startY;
      const next = Math.min(MAX_QUESTION_SIZE, Math.max(MIN_QUESTION_SIZE, startSize + dy));
      document.documentElement.style.setProperty('--question-size', `${next}px`);
    };
    const onUp = () => {
      this._questionResizer.classList.remove('dragging');
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);
      this._saveLayout({ questionSize: this._readCssVarPx('--question-size', 88) });
    };

    this._questionResizer.addEventListener('pointerdown', (e) => {
      startY = e.clientY;
      startSize = this._readCssVarPx('--question-size', 88);
      this._questionResizer.classList.add('dragging');
      document.addEventListener('pointermove', onMove);
      document.addEventListener('pointerup', onUp);
    });
  }

  /** @private Press-and-hold on the camera or emotion panel, then drag sideways, to swap their sides. */
  _initSwapDrag() {
    for (const panel of [this._videoPanel, this._emotionPanelEl]) {
      let pressTimer = null;
      let armed = false;
      let startX = 0;

      const cleanup = () => {
        clearTimeout(pressTimer);
        armed = false;
        panel.classList.remove('swap-armed');
        document.removeEventListener('pointermove', onMove);
        document.removeEventListener('pointerup', onUp);
      };
      const onMove = (e) => {
        if (!armed || Math.abs(e.clientX - startX) < SWAP_DRAG_THRESHOLD_PX) return;
        this._mainContent.classList.toggle('layout-swapped');
        this._saveLayout({ swapped: this._mainContent.classList.contains('layout-swapped') });
        cleanup();
      };
      const onUp = () => cleanup();

      panel.addEventListener('pointerdown', (e) => {
        startX = e.clientX;
        pressTimer = setTimeout(() => {
          armed = true;
          panel.classList.add('swap-armed');
          document.addEventListener('pointermove', onMove);
          document.addEventListener('pointerup', onUp);
        }, LONG_PRESS_MS);
      });
      panel.addEventListener('pointerup', () => clearTimeout(pressTimer));
      panel.addEventListener('pointerleave', () => { if (!armed) clearTimeout(pressTimer); });
    }
  }

  /** @private Apply a layout config to the DOM via CSS custom properties + the swap class. */
  _applyLayout(layout) {
    document.documentElement.style.setProperty('--camera-size',   `${layout.cameraSize}px`);
    document.documentElement.style.setProperty('--question-size', `${layout.questionSize}px`);
    this._mainContent.classList.toggle('layout-swapped', !!layout.swapped);
  }

  /** @private Read the persisted layout from localStorage, falling back to defaults. */
  _loadLayout() {
    const defaults = { cameraSize: 340, questionSize: 88, swapped: false };
    try {
      const stored = JSON.parse(localStorage.getItem(LAYOUT_STORAGE_KEY));
      return { ...defaults, ...stored };
    } catch {
      return defaults;
    }
  }

  /** @private Merge a partial layout change into the persisted layout in localStorage. */
  _saveLayout(partial) {
    const merged = { ...this._loadLayout(), ...partial };
    localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(merged));
  }

  /** @private Read a px-valued CSS custom property off the root element. */
  _readCssVarPx(name, fallback) {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(name);
    const value = parseInt(raw, 10);
    return Number.isFinite(value) ? value : fallback;
  }

  /**
   * @private
   * @param {string} state - one of idle/connecting/active/ended/error, matches a CSS class suffix.
   * @param {string} label - visible badge text.
   */
  _setStatus(state, label) {
    this._statusBadge.className = `status-badge status-${state}`;
    this._statusBadge.textContent = label;
  }

  /** @private Start the mm:ss session clock, ticking once per second. */
  _startTimer() {
    this._timerStart = Date.now();
    this._timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - this._timerStart) / 1000);
      const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const ss = String(elapsed % 60).padStart(2, '0');
      this._sessionTimer.textContent = `${mm}:${ss}`;
    }, 1000);
  }

  /** @private Stop the session clock, if running. */
  _stopTimer() {
    if (this._timerInterval) {
      clearInterval(this._timerInterval);
      this._timerInterval = null;
    }
  }

  /** @private Clear the error message area. */
  _clearError() {
    this._errorMessage.textContent = '';
  }

  /**
   * @private Append one transcript line and auto-scroll the feed to it.
   * @param {string} text
   */
  _appendTranscriptLine(text) {
    const span = document.createElement('span');
    span.className   = 'transcript-segment';
    span.textContent = text;
    this._transcriptFeed.appendChild(span);
    // Auto-scroll to latest entry
    this._transcriptFeed.scrollTop = this._transcriptFeed.scrollHeight;
  }
}
