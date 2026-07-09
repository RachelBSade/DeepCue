/**
 * i18n.js
 *
 * Minimal UI translation layer. Language choice persists in localStorage
 * so it carries from the landing page to the interview page.
 *
 * Usage: give any element a `data-i18n="key"` attribute; applyTranslations()
 * sets its textContent from the dictionary below. Also flips <html> dir/lang.
 */

const STORAGE_KEY = 'deepcue_lang';

const STRINGS = {
  en: {
    'landing.title':        'Welcome to',
    'landing.subtitle':     "Let's get a few details before we begin.",
    'landing.greeting':     "Hi! I'm here to guide you through your interview.",
    'landing.name_label':   'Full name',
    'landing.email_label':  'Email',
    'landing.optional':     '(optional)',
    'landing.email_hint':   "Leave your email if you'd like your interview report sent to you automatically once it's ready.",
    'landing.name_ph':      'Your full name',
    'landing.submit':       'Continue →',
    'header.timer_label':   'Idle',
    'panel.emotion_title':  'Emotion Analysis',
    'panel.dominant':       'Dominant',
    'panel.transcript':     'Transcript',
    'panel.transcript_lang':'(Hebrew)',
    'controls.start':       '▶ Start Interview',
    'controls.stop':        '■ Stop',
    'question.label':       'Question',
    'question.next':        'Next Question →',
    'question.done':        "That's the last question — click Stop when you're ready to finish.",
    'countdown.label':      'Get ready…',
    'lang.toggle':           'עב',
    'panel.speech_rate':    'Speech rate',
    'panel.speech_rate_unit': 'wpm',
  },
  he: {
    'landing.title':        'ברוכים הבאים ל',
    'landing.subtitle':     'בואו נמלא כמה פרטים לפני שנתחיל.',
    'landing.greeting':     'הי! אני כאן כדי להדריך אותך בראיון.',
    'landing.name_label':   'שם מלא',
    'landing.email_label':  'אימייל',
    'landing.optional':     '(אופציונלי)',
    'landing.email_hint':   'השאר/י אימייל אם תרצה/י שדוח הראיון יישלח אליך אוטומטית כשיהיה מוכן.',
    'landing.name_ph':      'השם המלא שלך',
    'landing.submit':       '← המשך',
    'header.timer_label':   'במנוחה',
    'panel.emotion_title':  'ניתוח רגשות',
    'panel.dominant':       'רגש דומיננטי',
    'panel.transcript':     'תמלול',
    'panel.transcript_lang':'(עברית)',
    'controls.start':       '▶ התחל ראיון',
    'controls.stop':        '■ עצור',
    'question.label':       'שאלה',
    'question.next':        '← שאלה הבאה',
    'question.done':        'זו השאלה האחרונה — לחצ/י על "עצור" כשתסיים/י.',
    'countdown.label':      'התכוננ/י…',
    'lang.toggle':           'EN',
    'panel.speech_rate':    'קצב דיבור',
    'panel.speech_rate_unit': 'מילים/דקה',
  },
};

/** @returns {string} the persisted language code ('en' | 'he'), defaulting to 'en'. */
export function getLang() {
  return localStorage.getItem(STORAGE_KEY) || 'en';
}

/**
 * Persist the language choice and flip <html> lang/dir to match.
 * @param {string} lang - 'en' or 'he'.
 */
export function setLang(lang) {
  localStorage.setItem(STORAGE_KEY, lang);
  document.documentElement.lang = lang === 'he' ? 'he' : 'en';
  document.documentElement.dir  = lang === 'he' ? 'rtl' : 'ltr';
}

/**
 * Translate a dictionary key in the current language, falling back to English then the key itself.
 * @param {string} key
 * @returns {string}
 */
export function t(key) {
  const lang = getLang();
  return STRINGS[lang]?.[key] ?? STRINGS.en[key] ?? key;
}

/** Apply translations to every element with a data-i18n attribute, and fix dir/lang. */
export function applyTranslations() {
  const lang = getLang();
  document.documentElement.lang = lang === 'he' ? 'he' : 'en';
  document.documentElement.dir  = lang === 'he' ? 'rtl' : 'ltr';

  document.querySelectorAll('[data-i18n]').forEach((el) => {
    el.textContent = t(el.getAttribute('data-i18n'));
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
    el.setAttribute('placeholder', t(el.getAttribute('data-i18n-placeholder')));
  });
}

/** Toggle between 'en' and 'he', persist, and re-apply translations. */
export function toggleLang() {
  setLang(getLang() === 'he' ? 'en' : 'he');
  applyTranslations();
}
