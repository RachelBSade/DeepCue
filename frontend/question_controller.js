/**
 * question_controller.js
 *
 * "Guided Question Carousel" — shows one HR interview question at a time.
 * Self-paced: the candidate clicks "Next Question" when ready to move on,
 * rather than a forced timer. Purely client-side; not sent to the backend.
 */

import { t, getLang } from './i18n.js';
import { getQuestions as getQuestionsForLang } from './questions.js';

/** @returns {string[]} the question list in whatever language is currently active. */
function getQuestions() {
  return getQuestionsForLang(getLang());
}

const COUNTDOWN_SECONDS = 5;

export class QuestionController {
  constructor() {
    this._panel          = document.getElementById('question-panel');
    this._progressEl     = document.getElementById('question-progress');
    this._textEl         = document.getElementById('question-text');
    this._nextBtn        = document.getElementById('question-next');
    this._countdownEl    = document.getElementById('countdown-overlay');
    this._countdownNumEl = document.getElementById('countdown-number');
    this._index          = 0;
    this._questions      = [];

    this._nextBtn.addEventListener('click', () => this._advance());
  }

  /** Run a 5-second countdown, then show the panel with the first question. */
  start() {
    this._questions = getQuestions();
    this._index = 0;

    this._countdownEl.classList.remove('hidden');
    let remaining = COUNTDOWN_SECONDS;
    this._tickCountdown(remaining);

    const interval = setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(interval);
        this._countdownEl.classList.add('hidden');
        this._panel.classList.remove('hidden');
        this._render();
        return;
      }
      this._tickCountdown(remaining);
    }, 1000);
  }

  /** @private Update the big number and restart its pulse animation. */
  _tickCountdown(n) {
    this._countdownNumEl.textContent = String(n);
    this._countdownNumEl.style.animation = 'none';
    // Force reflow so the animation restarts on the next tick.
    void this._countdownNumEl.offsetWidth;
    this._countdownNumEl.style.animation = '';
  }

  /** Hide the panel and countdown overlay (e.g. when the session ends). */
  stop() {
    this._panel.classList.add('hidden');
    this._countdownEl.classList.add('hidden');
  }

  /** Re-translate the current question (and labels) after a language switch, no index reset. */
  refreshLanguage() {
    if (!this._questions.length) return;
    this._questions = getQuestions();
    const isDone = this._nextBtn.disabled;
    if (isDone) {
      this._progressEl.textContent = `${t('question.label')} ${this._questions.length}/${this._questions.length}`;
      this._textEl.textContent = t('question.done');
    } else {
      this._render();
    }
  }

  /** @private Move to the next question, or show the "done" state if already on the last one. */
  _advance() {
    if (this._index < this._questions.length - 1) {
      this._index += 1;
      this._render();
    } else {
      this._textEl.textContent = t('question.done');
      this._nextBtn.disabled = true;
    }
  }

  /** @private Paint the current index/question/button text into the DOM. */
  _render() {
    this._progressEl.textContent = `${t('question.label')} ${this._index + 1}/${this._questions.length}`;
    this._textEl.textContent = this._questions[this._index];
    this._nextBtn.disabled = false;
    this._nextBtn.textContent = t('question.next');
  }
}
