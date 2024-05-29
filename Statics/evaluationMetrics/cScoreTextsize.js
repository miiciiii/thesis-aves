/**
 * Calculate the C-Score Textsize.
 * @param {number} correctAnswers - Percentage of correct answers (Pr).
 * @param {number} textsize - Text size (Ts).
 * @param {number} numQuestions - Number of questions (Nq).
 * @param {number[]} questionSizes - Array of question sizes (Qs) for each question.
 * @param {number[]} meanTimes - Array of mean times (t_mean) for each question (in seconds).
 * @returns {number} The C-Score Textsize.
 */
function calculateCScoreTextsize(correctAnswers, textsize, numQuestions, questionSizes, meanTimes) {
  // Calculate the sum of Qs(q) / t_mean(q) for all questions
  const sumOfRatios = questionSizes.reduce((sum, size, q) => {
    const ratio = size / meanTimes[q];
    return sum + ratio;
  }, 0);

  // C-Score Textsize formula
  return (correctAnswers * textsize) / numQuestions * sumOfRatios;
}

// Example usage:
const pr = 0.8; // Example percentage of correct answers
const ts = 100; // Example text size
const nq = 5; // Example number of questions
const qsArray = [10, 15, 12, 18, 20]; // Example question sizes for each question
const meanTimesArray = [4, 5, 4, 5, 4]; // Example mean times for each question (in seconds)
const cScoreTextsize = calculateCScoreTextsize(pr, ts, nq, qsArray, meanTimesArray);
console.log(`C-Score Textsize: ${cScoreTextsize}`);
