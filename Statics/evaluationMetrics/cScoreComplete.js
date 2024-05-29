/**
 * Calculate the C-Complete score.
 * @param {number} correctAnswers - Percentage of correct answers (Pr).
 * @param {number} numQuestions - Number of questions (Nq).
 * @param {number[]} questionSizes - Array of question sizes (Qs) for each question.
 * @param {number[]} meanTimes - Array of mean times (t_mean) for each question (in seconds).
 * @returns {number} The C-Complete score.
 */
function calculateCComplete(correctAnswers, numQuestions, questionSizes, meanTimes) {
  // Calculate the sum of Qs(q) / t_mean(q) for all questions
  const sumOfRatios = questionSizes.reduce((sum, size, q) => {
    const ratio = size / meanTimes[q];
    return sum + ratio;
  }, 0);

  // Complete C-Score formula
  return (correctAnswers / numQuestions) * sumOfRatios;
}

// Example usage:
const pr = 0.8; // Example percentage of correct answers
const nq = 5; // Example number of questions
const qsArray = [10, 15, 12, 18, 20]; // Example question sizes for each question
const meanTimesArray = [4, 5, 4, 5, 4]; // Example mean times for each question (in seconds)
const cScoreComplete = calculateCComplete(pr, nq, qsArray, meanTimesArray);
console.log(`C-Score Complete: ${cScoreComplete}`);