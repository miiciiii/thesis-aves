document.addEventListener('DOMContentLoaded', function () {
    let startTime = new Date().getTime();
    let readingMode = 'Quiet Reading';
    
    let currentQuizIndex = 0; // To keep track of the current quiz item
    let quizData; // To store the quiz data
    let userResponses = []; // To store user responses
    let questionStartTime = new Date().getTime(); // Start time for the current question
    let timeToRespond = []; // To store the time taken to respond to each question

    const quizRecord = {
        "date": '',
        "duration": '',
        "mode": readingMode,
        "correct-answer": [],
        "responses": [],
        "timeToRespond": timeToRespond
    };

    function getDate() {
        const currentDate = new Date();
        const date = currentDate.getDate();
        const month = currentDate.getMonth() + 1;
        const year = currentDate.getFullYear();
        const hours = currentDate.getHours();
        const minutes = currentDate.getMinutes();
        const seconds = currentDate.getSeconds();

        const formattedDate = `${year}-${month}-${date} ${hours}:${minutes}:${seconds}`;
        return formattedDate;
    }

    function getDuration() {
        const endTime = new Date().getTime();
        const duration = endTime - startTime;

        const hours = Math.floor(duration / 3600000);
        const minutes = Math.floor((duration % 3600000) / 60000);
        const seconds = Math.floor((duration % 60000) / 1000);

        const formattedDuration = `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
        return formattedDuration;
    }

    function loadQuizData() {
        fetch('quiz_data.json')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                quizData = data;
                userResponses = new Array(quizData.length).fill(""); // Initialize user responses with empty strings
                quizRecord.timeToRespond = new Array(quizData.length).fill(0); // Initialize response times with 0
                displayQuizItem(currentQuizIndex); // Display the first quiz item
            })
            .catch(error => console.error('Error loading quiz data:', error));
    }

    function displayQuizItem(index) {
        const quizItem = quizData[index];
        const passage = quizItem.passage;
        const question = quizItem["questions-choices-answer"].question;
        const choices = quizItem["questions-choices-answer"].choices;

        document.getElementById("passage").textContent = passage;
        document.getElementById("question").textContent = question;

        const choicesDiv = document.getElementById("choices-div");
        choicesDiv.innerHTML = ""; // Clear previous choices

        choices.forEach((choice, i) => {
            const label = document.createElement("label");
            label.classList.add("choice-label");
            const input = document.createElement("input");
            input.type = "radio";
            input.name = "choice";
            input.value = choice;
            input.id = `choice${i}`;

            // Check if this choice was previously selected
            if (userResponses[index] === choice) {
                input.checked = true;
                label.classList.add('selected');
            }

            // Event listener to change background color when radio button is selected
            input.addEventListener('change', () => {
                document.querySelectorAll('.choice-label').forEach(label => {
                    label.classList.remove('selected');
                });
                if (input.checked) {
                    label.classList.add('selected');
                }
                // Save user's choice
                userResponses[index] = choice;
            });

            label.appendChild(input);
            label.appendChild(document.createTextNode(choice));
            choicesDiv.appendChild(label);
            choicesDiv.appendChild(document.createElement("br"));
        });

        // Handle submit button visibility
        const submitButton = document.getElementById("submit-button");
        if (index === quizData.length - 1) {
            submitButton.style.display = 'block'; // Show the submit button on the last question
        } else {
            submitButton.style.display = 'none'; // Hide the submit button otherwise
        }

        // Reset question start time
        questionStartTime = new Date().getTime();
    }

    function handleNavigation(direction) {
        const currentTime = new Date().getTime();
        const timeSpent = currentTime - questionStartTime;
        quizRecord.timeToRespond[currentQuizIndex] += timeSpent;

        const selectedChoice = document.querySelector('input[name="choice"]:checked');
        if (selectedChoice) {
            userResponses[currentQuizIndex] = selectedChoice.value; // Set answer
        } else {
            userResponses[currentQuizIndex] = ""; // Save blank if no choice is selected
        }

        if (direction === 'next') {
            if (currentQuizIndex < quizData.length - 1) {
                currentQuizIndex++;
                displayQuizItem(currentQuizIndex);
            } else {
                alert("You are already at the last quiz item.");
            }
        } else if (direction === 'previous') {
            if (currentQuizIndex > 0) {
                currentQuizIndex--;
                displayQuizItem(currentQuizIndex);
            } else {
                alert("You are already at the first quiz item.");
            }
        }
    }

    document.getElementById("previous-button").addEventListener("click", function() {
        handleNavigation('previous');
    });

    document.getElementById("next-button").addEventListener("click", function() {
        handleNavigation('next');
    });

    document.getElementById("submit-button").addEventListener("click", function() {
        submitQuiz();
    });

    function submitQuiz() {
        const currentTime = new Date().getTime();
        const timeSpent = currentTime - questionStartTime;
        quizRecord.timeToRespond[currentQuizIndex] += timeSpent;

        const selectedChoice = document.querySelector('input[name="choice"]:checked');
        if (selectedChoice) {
            userResponses[currentQuizIndex] = selectedChoice.value;
        } else {
            userResponses[currentQuizIndex] = ""; // Save blank if no choice is selected
        }

        quizData.forEach((quizItem, index) => {
            const correctAnswer = quizItem["questions-choices-answer"].answer;
            quizRecord["correct-answer"].push(correctAnswer);
            quizRecord.responses.push(userResponses[index]);
        });

        alert("End of quiz.");

        quizRecord.date = getDate();
        quizRecord.duration = getDuration();

        localStorage.clear();
        localStorage.setItem('quizRecord', JSON.stringify(quizRecord));

        window.location.href = 'result.html';
    }

    loadQuizData();
});