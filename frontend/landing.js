/**
 * landing.js
 *
 * Entrance form: collects candidate name (required) + email (optional),
 * stores them in sessionStorage, then navigates to the interview screen.
 */

import { applyTranslations, toggleLang } from './i18n.js';

const form       = document.getElementById('landing-form');
const nameInput  = document.getElementById('full-name-input');
const emailInput = document.getElementById('email-input');
const errorEl    = document.getElementById('landing-error');

applyTranslations();
document.getElementById('lang-toggle').addEventListener('click', toggleLang);

form.addEventListener('submit', (event) => {
  event.preventDefault();

  const name  = nameInput.value.trim();
  const email = emailInput.value.trim();

  if (!name) {
    errorEl.textContent = 'Please enter your full name.';
    return;
  }
  if (email && !_isValidEmail(email)) {
    errorEl.textContent = 'Please enter a valid email address, or leave it blank.';
    return;
  }

  sessionStorage.setItem('deepcue_candidate_name', name);
  sessionStorage.setItem('deepcue_candidate_email', email);

  window.location.href = 'interview.html';
});

/**
 * Loose client-side email shape check (server does the real validation).
 * @param {string} value
 * @returns {boolean}
 */
function _isValidEmail(value) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}
