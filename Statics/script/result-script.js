document.addEventListener('DOMContentLoaded', function () {
    let currentQuizIndex = 0;
    let quizData;
    let quizRecord;

    let correctBorderColor = "#70a83b";
    let incorrectBorderColor = "#ff5858 ";
    let otherChoiceBorderColor = "none";
    let unchoiceBackgroundColor = "#566799";

    let correctResponses = 0;
    let numberOfQuestions = 0;
    let averageTimeToRespond = 0;

    // Retrieve quizRecord from local storage
    const quizRecordString = localStorage.getItem('quizRecord');

    if (quizRecordString) {
        quizRecord = JSON.parse(quizRecordString);
        displayRecord();
        loadQuizData();
        updateTable(quizRecord);
    } else {
        alert('No quiz record found in local storage.');
    }


    function displayRecord() {
        document.getElementById('date').textContent = quizRecord.date;
        document.getElementById('duration').textContent = quizRecord.duration;
        document.getElementById('reading-mode').textContent = quizRecord.mode;
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
                displayQuizItem(currentQuizIndex); // Display the first quiz item
                populateGlobalVariables();
            })
            .catch(error => console.error('Error loading quiz data:', error));
    }

    function populateGlobalVariables(){
        numberOfQuestions = quizData.length;
        averageTimeToRespond = calculateAverageTime();
    }

    function displayQuizItem(index) {
        const quizItem = quizData[index];
        const passage = quizItem.passage;
        const question = quizItem["questions-choices-answer"].question;
        const choices = quizItem["questions-choices-answer"].choices;
        const correctAnswer = quizItem["questions-choices-answer"].answer;
        const userResponse = quizRecord.responses[index];

        document.getElementById("passage").textContent = passage;
        document.getElementById("question").textContent = question;

        const choicesDiv = document.getElementById("choices-div");
        choicesDiv.innerHTML = ""; // Clear previous choices

        choices.forEach(choice => {
            const choiceText = document.createElement("p");
            choiceText.classList.add("choice-label");
            choiceText.textContent = choice;
            
            // Changes the border of the choices
            choiceText.style.border = userResponse === choice ? 
                              (choice === correctAnswer ? `3px solid ${correctBorderColor}` : `3px solid ${incorrectBorderColor}`) : 
                              (choice === correctAnswer ? `3px solid ${correctBorderColor}` : `3px solid ${otherChoiceBorderColor}`);
            choiceText.style.backgroundColor = choice !== correctAnswer ? unchoiceBackgroundColor : "none";
            
            choicesDiv.appendChild(choiceText);
        });
    }

    function handleNavigation(direction) {
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

    function convertToSeconds(millisecondsArray) {
        return millisecondsArray.map(milliseconds => milliseconds / 1000);
    }

    function updateTable(quizRecord) {
        const correctAnswers = quizRecord["correct-answer"];
        const responses = quizRecord.responses;
        const durations = convertToSeconds(quizRecord.timeToRespond);

        const table = document.getElementById("remarks-table");
        
        // Clear existing table rows except the header
        table.innerHTML = '<tr><th>Question</th><th>Remarks</th><th>Duration</th></tr>';

        let questionNumber = 1;

        correctAnswers.forEach((correctAnswer, index) => {
            const response = responses[index];
            const row = document.createElement('tr');
            const questionNumCell = document.createElement('td');
            const remarksCell = document.createElement('td');
            const durationCell = document.createElement('td');

            questionNumCell.textContent = questionNumber++;

            let remark;
            if (response === correctAnswer){
                remark = "Correct";
                correctResponses++;
            } else if (response === ""){
                remark = "Wrong (No Response)";
            } else {
                remark = "Wrong";
            }

            numberOfQuestions++;

            remarksCell.textContent = remark;
            let duration = durations[index];
            durationCell.textContent = `${duration}s`; // Display duration for each question

            row.appendChild(questionNumCell);
            row.appendChild(remarksCell);
            row.appendChild(durationCell);

            table.appendChild(row);
        });
    }

    function calculateAverageTime(timeArray = convertToSeconds(quizRecord.timeToRespond)) {
        // Check if the array is empty
        if (timeArray.length === 0) {
            return 0; // Return 0 if array is empty
        }
        
        // Sum up all the time values in the array
        const sum = timeArray.reduce((total, time) => total + time, 0);
        
        // Calculate the average by dividing the sum by the number of elements
        const average = sum / timeArray.length;
        
        return average;
    }

     // Calculate necessary values
    const correctPercentage = (correctResponses / numberOfQuestions) * 100;
    const averageTime = calculateAverageTime();
    const score = Number(calculateCSimple(correctPercentage, averageTime).toFixed(3)); // Round off to 3 decimal places
    document.getElementById('overall-score').textContent = score;

    alert(correctPercentage)
    alert(averageTime)
    alert(score)

    loadQuizData();
    updateTable(quizRecord);
});
