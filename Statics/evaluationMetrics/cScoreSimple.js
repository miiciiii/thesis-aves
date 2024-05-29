/**
 * Calculate the C-Score Simple.
 * @param {number} correctAnswers - Percentage of correct answers (Pr).
 * @param {number} averageTime - Average time (tmean).
 * @returns {number} The C-Score Simple.
 */

function calculateCSimple(correctAnswers, averageTime) {
  // Simple C-Score formula
  return correctAnswers / averageTime;
}