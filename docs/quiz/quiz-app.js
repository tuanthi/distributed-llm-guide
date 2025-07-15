// Production ML Engineering Quiz Application
class QuizApp {
    constructor() {
        this.currentQuestion = 0;
        this.userAnswers = [];
        this.score = 0;
        this.startTime = null;
        this.endTime = null;
        this.randomizedQuestions = [];
        this.isAnswered = false;
        this.hasShownFeedback = false;
        
        this.initializeApp();
    }
    
    initializeApp() {
        this.bindEvents();
        this.showScreen('welcomeScreen');
    }
    
    bindEvents() {
        // Start quiz button
        document.getElementById('startQuiz').addEventListener('click', () => {
            this.startQuiz();
        });
        
        // Navigation buttons
        document.getElementById('prevBtn').addEventListener('click', () => {
            this.previousQuestion();
        });
        
        document.getElementById('nextBtn').addEventListener('click', () => {
            if (this.isAnswered && !this.hasShownFeedback) {
                this.showFeedback();
            } else {
                this.nextQuestion();
            }
        });
        
        // Results screen buttons
        document.getElementById('reviewAnswers').addEventListener('click', () => {
            this.showReview();
        });
        
        document.getElementById('retakeQuiz').addEventListener('click', () => {
            this.resetQuiz();
        });
        
        document.getElementById('backToResults').addEventListener('click', () => {
            this.showScreen('resultsScreen');
        });
    }
    
    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }
    
    startQuiz() {
        this.startTime = new Date();
        this.currentQuestion = 0;
        
        // Randomize questions
        this.randomizedQuestions = this.shuffleArray([...quizData]);
        this.userAnswers = new Array(this.randomizedQuestions.length).fill(null);
        this.score = 0;
        this.isAnswered = false;
        this.hasShownFeedback = false;
        
        this.showScreen('quizScreen');
        this.displayQuestion();
        this.updateProgress();
    }
    
    displayQuestion() {
        const question = this.randomizedQuestions[this.currentQuestion];
        this.isAnswered = false;
        this.hasShownFeedback = false;
        
        // Update question header
        document.getElementById('questionNumber').textContent = `Question ${this.currentQuestion + 1}`;
        document.getElementById('questionType').textContent = 
            question.type === 'multiple-choice' ? 'Multiple Choice' : 'Short Answer';
        
        // Update question content
        document.getElementById('questionText').textContent = question.question;
        
        // Handle code block
        const codeBlock = document.getElementById('questionCode');
        if (question.code) {
            codeBlock.innerHTML = `<pre><code class="language-python">${question.code}</code></pre>`;
            codeBlock.classList.remove('hidden');
            // Re-highlight syntax
            if (typeof Prism !== 'undefined') {
                Prism.highlightElement(codeBlock.querySelector('code'));
            }
        } else {
            codeBlock.classList.add('hidden');
        }
        
        // Clear answer options and feedback
        const answerContainer = document.getElementById('answerOptions');
        answerContainer.innerHTML = '';
        
        // Clear any existing feedback
        const existingFeedback = document.getElementById('feedbackContainer');
        if (existingFeedback) {
            existingFeedback.remove();
        }
        
        if (question.type === 'multiple-choice') {
            question.options.forEach((option, index) => {
                const optionDiv = document.createElement('div');
                optionDiv.className = 'answer-option';
                optionDiv.innerHTML = `
                    <input type="radio" id="option${index}" name="answer" value="${index}">
                    <label for="option${index}">${option}</label>
                `;
                answerContainer.appendChild(optionDiv);
                
                // Add click handler
                optionDiv.addEventListener('click', () => {
                    if (!this.hasShownFeedback) {
                        document.getElementById(`option${index}`).checked = true;
                        this.selectAnswer(index);
                    }
                });
            });
        } else {
            // Short answer input
            const inputDiv = document.createElement('div');
            inputDiv.className = 'answer-input';
            inputDiv.innerHTML = `
                <textarea id="shortAnswer" placeholder="Enter your answer here..." rows="3" ${this.hasShownFeedback ? 'disabled' : ''}></textarea>
            `;
            answerContainer.appendChild(inputDiv);
            
            const textarea = document.getElementById('shortAnswer');
            textarea.addEventListener('input', () => {
                if (!this.hasShownFeedback) {
                    this.selectAnswer(textarea.value.trim());
                }
            });
        }
        
        // Restore previous answer if exists
        if (this.userAnswers[this.currentQuestion] !== null) {
            this.restoreAnswer();
        }
        
        // Update navigation buttons
        this.updateNavigationButtons();
    }
    
    selectAnswer(answer) {
        this.userAnswers[this.currentQuestion] = answer;
        this.isAnswered = true;
        this.updateNavigationButtons();
    }
    
    restoreAnswer() {
        const answer = this.userAnswers[this.currentQuestion];
        const question = this.randomizedQuestions[this.currentQuestion];
        
        if (question.type === 'multiple-choice') {
            const radio = document.querySelector(`input[value="${answer}"]`);
            if (radio) radio.checked = true;
        } else {
            const textarea = document.getElementById('shortAnswer');
            if (textarea) textarea.value = answer;
        }
    }
    
    updateNavigationButtons() {
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        
        // Previous button
        prevBtn.disabled = this.currentQuestion === 0;
        
        // Next button logic
        const hasAnswer = this.userAnswers[this.currentQuestion] !== null;
        
        if (!this.isAnswered) {
            nextBtn.disabled = true;
            nextBtn.innerHTML = 'Select Answer First';
        } else if (this.isAnswered && !this.hasShownFeedback) {
            nextBtn.disabled = false;
            nextBtn.innerHTML = 'Submit Answer <i class="fas fa-check"></i>';
        } else {
            nextBtn.disabled = false;
            if (this.currentQuestion === this.randomizedQuestions.length - 1) {
                nextBtn.innerHTML = 'Finish Quiz <i class="fas fa-flag-checkered"></i>';
            } else {
                nextBtn.innerHTML = 'Next Question <i class="fas fa-arrow-right"></i>';
            }
        }
    }
    
    showFeedback() {
        const question = this.randomizedQuestions[this.currentQuestion];
        const userAnswer = this.userAnswers[this.currentQuestion];
        let isCorrect = false;
        let userAnswerText = '';
        let correctAnswerText = '';
        
        // Check if answer is correct
        if (question.type === 'multiple-choice') {
            isCorrect = userAnswer === question.correct;
            userAnswerText = question.options[userAnswer];
            correctAnswerText = question.options[question.correct];
        } else {
            const userAnswerNorm = String(userAnswer).toLowerCase().trim();
            const correctAnswerNorm = String(question.correct).toLowerCase().trim();
            isCorrect = userAnswerNorm === correctAnswerNorm;
            userAnswerText = userAnswer;
            correctAnswerText = question.correct;
        }
        
        // Update score immediately
        if (isCorrect) {
            this.score++;
        }
        
        // Mark options as correct/incorrect for multiple choice
        if (question.type === 'multiple-choice') {
            document.querySelectorAll('.answer-option').forEach((option, index) => {
                option.style.pointerEvents = 'none';
                if (index === question.correct) {
                    option.classList.add('feedback-correct');
                } else if (index === userAnswer) {
                    option.classList.add('feedback-incorrect');
                } else {
                    option.classList.add('feedback-neutral');
                }
            });
        } else {
            document.getElementById('shortAnswer').disabled = true;
        }
        
        // Create feedback container
        const feedbackContainer = document.createElement('div');
        feedbackContainer.id = 'feedbackContainer';
        feedbackContainer.className = `feedback-container ${isCorrect ? 'feedback-correct-bg' : 'feedback-incorrect-bg'}`;
        
        feedbackContainer.innerHTML = `
            <div class="feedback-header">
                <span class="feedback-icon">
                    <i class="fas fa-${isCorrect ? 'check-circle' : 'times-circle'}"></i>
                </span>
                <span class="feedback-title">${isCorrect ? 'Correct!' : 'Incorrect'}</span>
                <span class="current-score">Score: ${this.score}/${this.currentQuestion + 1}</span>
            </div>
            
            ${!isCorrect ? `
                <div class="feedback-answer">
                    <strong>Your answer:</strong> ${userAnswerText}<br>
                    <strong>Correct answer:</strong> ${correctAnswerText}
                </div>
            ` : ''}
            
            <div class="feedback-explanation">
                <strong>Explanation:</strong>
                <p>${question.explanation}</p>
            </div>
        `;
        
        // Insert feedback after answer options
        const answerContainer = document.getElementById('answerOptions');
        answerContainer.parentNode.insertBefore(feedbackContainer, answerContainer.nextSibling);
        
        this.hasShownFeedback = true;
        this.updateNavigationButtons();
        this.updateProgress(); // Update progress with current score
    }
    
    previousQuestion() {
        if (this.currentQuestion > 0) {
            this.currentQuestion--;
            this.displayQuestion();
            this.updateProgress();
        }
    }
    
    nextQuestion() {
        if (this.currentQuestion < this.randomizedQuestions.length - 1) {
            this.currentQuestion++;
            this.displayQuestion();
            this.updateProgress();
        } else {
            this.finishQuiz();
        }
    }
    
    updateProgress() {
        const progress = ((this.currentQuestion + 1) / this.randomizedQuestions.length) * 100;
        document.getElementById('progressFill').style.width = `${progress}%`;
        document.getElementById('progressText').textContent = 
            `Question ${this.currentQuestion + 1} of ${this.randomizedQuestions.length} | Score: ${this.score}/${this.currentQuestion + (this.hasShownFeedback ? 1 : 0)}`;
    }
    
    finishQuiz() {
        this.endTime = new Date();
        // Score is already calculated during feedback, no need to recalculate
        this.showResults();
        this.showScreen('resultsScreen');
    }
    
    showResults() {
        // Update score display
        document.getElementById('finalScore').textContent = this.score;
        
        // Update score message
        const percentage = (this.score / this.randomizedQuestions.length) * 100;
        const scoreTitle = document.getElementById('scoreTitle');
        const scoreMessage = document.getElementById('scoreMessage');
        
        if (percentage >= 90) {
            scoreTitle.textContent = 'Outstanding! 🏆';
            scoreMessage.textContent = 'You have mastered production ML engineering concepts!';
        } else if (percentage >= 80) {
            scoreTitle.textContent = 'Excellent Work! 🌟';
            scoreMessage.textContent = 'You have a solid understanding of production ML engineering.';
        } else if (percentage >= 70) {
            scoreTitle.textContent = 'Good Job! 👍';
            scoreMessage.textContent = 'You\'re on the right track. Review the handbook for areas to improve.';
        } else if (percentage >= 60) {
            scoreTitle.textContent = 'Keep Learning! 📚';
            scoreMessage.textContent = 'Consider reviewing the handbook sections you found challenging.';
        } else {
            scoreTitle.textContent = 'More Study Needed 💪';
            scoreMessage.textContent = 'The handbook has all the information you need to master these concepts.';
        }
        
        // Generate breakdown by category
        this.generateResultsBreakdown();
    }
    
    generateResultsBreakdown() {
        const breakdown = {};
        
        // Group questions by category
        this.randomizedQuestions.forEach((question, index) => {
            const category = question.category;
            if (!breakdown[category]) {
                breakdown[category] = { correct: 0, total: 0 };
            }
            breakdown[category].total++;
            
            const userAnswer = this.userAnswers[index];
            let isCorrect = false;
            
            if (question.type === 'multiple-choice') {
                isCorrect = userAnswer === question.correct;
            } else {
                const userAnswerNorm = String(userAnswer).toLowerCase().trim();
                const correctAnswerNorm = String(question.correct).toLowerCase().trim();
                isCorrect = userAnswerNorm === correctAnswerNorm;
            }
            
            if (isCorrect) breakdown[category].correct++;
        });
        
        // Display breakdown
        const breakdownGrid = document.getElementById('breakdownGrid');
        breakdownGrid.innerHTML = '';
        
        Object.entries(breakdown).forEach(([category, stats]) => {
            const percentage = (stats.correct / stats.total) * 100;
            const item = document.createElement('div');
            item.className = 'breakdown-item';
            item.innerHTML = `
                <div class="category-name">${category}</div>
                <div class="category-score">${stats.correct}/${stats.total}</div>
                <div class="category-bar">
                    <div class="category-fill" style="width: ${percentage}%"></div>
                </div>
            `;
            breakdownGrid.appendChild(item);
        });
    }
    
    showReview() {
        this.generateReviewContent();
        this.showScreen('reviewScreen');
    }
    
    generateReviewContent() {
        const reviewContent = document.getElementById('reviewContent');
        reviewContent.innerHTML = '';
        
        this.randomizedQuestions.forEach((question, index) => {
            const userAnswer = this.userAnswers[index];
            let isCorrect = false;
            let userAnswerText = '';
            let correctAnswerText = '';
            
            if (question.type === 'multiple-choice') {
                isCorrect = userAnswer === question.correct;
                userAnswerText = userAnswer !== null ? question.options[userAnswer] : 'No answer';
                correctAnswerText = question.options[question.correct];
            } else {
                const userAnswerNorm = String(userAnswer).toLowerCase().trim();
                const correctAnswerNorm = String(question.correct).toLowerCase().trim();
                isCorrect = userAnswerNorm === correctAnswerNorm;
                userAnswerText = userAnswer || 'No answer';
                correctAnswerText = question.correct;
            }
            
            const reviewItem = document.createElement('div');
            reviewItem.className = `review-item ${isCorrect ? 'correct' : 'incorrect'}`;
            
            reviewItem.innerHTML = `
                <div class="review-header">
                    <span class="question-number">Question ${index + 1}</span>
                    <span class="question-category">${question.category}</span>
                    <span class="result-icon">
                        <i class="fas fa-${isCorrect ? 'check-circle' : 'times-circle'}"></i>
                    </span>
                </div>
                
                <div class="review-question">
                    <h4>${question.question}</h4>
                    ${question.code ? `<pre><code class="language-python">${question.code}</code></pre>` : ''}
                </div>
                
                <div class="review-answers">
                    <div class="user-answer ${isCorrect ? 'correct' : 'incorrect'}">
                        <strong>Your answer:</strong> ${userAnswerText}
                    </div>
                    ${!isCorrect ? `<div class="correct-answer">
                        <strong>Correct answer:</strong> ${correctAnswerText}
                    </div>` : ''}
                </div>
                
                <div class="review-explanation">
                    <strong>Explanation:</strong>
                    <p>${question.explanation}</p>
                </div>
            `;
            
            reviewContent.appendChild(reviewItem);
        });
        
        // Re-highlight code blocks
        if (typeof Prism !== 'undefined') {
            Prism.highlightAll();
        }
    }
    
    resetQuiz() {
        this.currentQuestion = 0;
        this.userAnswers = [];
        this.score = 0;
        this.startTime = null;
        this.endTime = null;
        this.randomizedQuestions = [];
        this.isAnswered = false;
        this.hasShownFeedback = false;
        this.showScreen('welcomeScreen');
    }
    
    showScreen(screenId) {
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        
        // Show target screen
        document.getElementById(screenId).classList.add('active');
    }
}

// Initialize quiz when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new QuizApp();
});