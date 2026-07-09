/**
 * questions.js
 *
 * The 10 most common HR interview questions, in English and Hebrew.
 * Indexes must line up 1:1 between the two arrays.
 */

export const QUESTIONS_EN = [
  'Tell me about yourself.',
  'Why do you want to work here?',
  'What are your greatest strengths?',
  'What is your greatest weakness?',
  'Why should we hire you?',
  'Where do you see yourself in five years?',
  'Why did you leave your last job?',
  'Describe a challenge you faced at work and how you handled it.',
  'What are your salary expectations?',
  'Do you have any questions for us?',
];

export const QUESTIONS_HE = [
  'ספר/י לי על עצמך.',
  'מדוע את/ה רוצה לעבוד כאן?',
  'מה החוזקות הבולטות שלך?',
  'מה החולשה הגדולה שלך?',
  'מדוע עלינו לשכור אותך?',
  'איפה את/ה רואה את עצמך בעוד חמש שנים?',
  'מדוע עזבת את העבודה הקודמת שלך?',
  'תאר/י אתגר שחווית בעבודה וכיצד התמודדת איתו.',
  'מה ציפיות השכר שלך?',
  'יש לך שאלות אלינו?',
];

/** @returns {string[]} the question list for the given language code ('en' | 'he') */
export function getQuestions(lang) {
  return lang === 'he' ? QUESTIONS_HE : QUESTIONS_EN;
}
