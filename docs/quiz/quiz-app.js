// Production ML Engineering Quiz Application
class QuizApp {
    constructor() {
        this.currentQuestion = 0;
        this.userAnswers = [];
        this.score = 0;
        this.startTime = null;
        this.endTime = null;
        
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
            this.nextQuestion();
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
    
    startQuiz() {
        this.startTime = new Date();
        this.currentQuestion = 0;
        this.userAnswers = new Array(quizData.length).fill(null);
        this.showScreen('quizScreen');
        this.displayQuestion();
        this.updateProgress();
    }
    
    displayQuestion() {
        const question = quizData[this.currentQuestion];
        
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
        
        // Clear and populate answer options
        const answerContainer = document.getElementById('answerOptions');
        answerContainer.innerHTML = '';
        
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
                    document.getElementById(`option${index}`).checked = true;
                    this.selectAnswer(index);
                });
            });
        } else {
            // Short answer input
            const inputDiv = document.createElement('div');
            inputDiv.className = 'answer-input';
            inputDiv.innerHTML = `
                <textarea id="shortAnswer" placeholder="Enter your answer here..." rows="3"></textarea>
            `;
            answerContainer.appendChild(inputDiv);
            
            const textarea = document.getElementById('shortAnswer');
            textarea.addEventListener('input', () => {
                this.selectAnswer(textarea.value.trim());
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
        this.updateNavigationButtons();
    }
    
    restoreAnswer() {
        const answer = this.userAnswers[this.currentQuestion];
        const question = quizData[this.currentQuestion];
        
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
        
        // Next button
        const hasAnswer = this.userAnswers[this.currentQuestion] !== null;
        nextBtn.disabled = !hasAnswer;
        
        // Update next button text
        if (this.currentQuestion === quizData.length - 1) {
            nextBtn.innerHTML = hasAnswer ? 'Finish Quiz <i class="fas fa-flag-checkered"></i>' : 'Finish Quiz';
        } else {
            nextBtn.innerHTML = 'Next <i class="fas fa-arrow-right"></i>';
        }
    }
    
    previousQuestion() {
        if (this.currentQuestion > 0) {
            this.currentQuestion--;
            this.displayQuestion();
            this.updateProgress();
        }
    }
    
    nextQuestion() {
        if (this.currentQuestion < quizData.length - 1) {
            this.currentQuestion++;
            this.displayQuestion();
            this.updateProgress();
        } else {
            this.finishQuiz();
        }
    }
    
    updateProgress() {
        const progress = ((this.currentQuestion + 1) / quizData.length) * 100;
        document.getElementById('progressFill').style.width = `${progress}%`;
        document.getElementById('progressText').textContent = 
            `Question ${this.currentQuestion + 1} of ${quizData.length}`;
    }
    
    finishQuiz() {
        this.endTime = new Date();
        this.calculateScore();
        this.showResults();
        this.showScreen('resultsScreen');
    }
    
    calculateScore() {
        this.score = 0;
        
        this.userAnswers.forEach((userAnswer, index) => {
            const question = quizData[index];
            let isCorrect = false;
            
            if (question.type === 'multiple-choice') {
                isCorrect = userAnswer === question.correct;
            } else {
                // Short answer - normalize and check
                const userAnswerNorm = String(userAnswer).toLowerCase().trim();
                const correctAnswerNorm = String(question.correct).toLowerCase().trim();
                isCorrect = userAnswerNorm === correctAnswerNorm;
            }
            
            if (isCorrect) this.score++;
        });
    }
    
    showResults() {
        // Update score display
        document.getElementById('finalScore').textContent = this.score;
        
        // Update score message
        const percentage = (this.score / quizData.length) * 100;
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
        quizData.forEach((question, index) => {
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
        
        quizData.forEach((question, index) => {
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