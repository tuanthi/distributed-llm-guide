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
        
        // Certificate button
        document.getElementById('getCertificate').addEventListener('click', () => {
            this.showCertificateForm();
        });
        
        // Generate certificate button
        document.getElementById('generateCertificate').addEventListener('click', () => {
            this.generateCertificate();
        });
        
        // Back to results from certificate
        document.getElementById('backToResultsFromCert').addEventListener('click', () => {
            this.showScreen('resultsScreen');
        });
        
        // Back to results from certificate display
        document.getElementById('backToResultsFromDisplay').addEventListener('click', () => {
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
        
        // Show/hide certificate button based on score
        const getCertificateBtn = document.getElementById('getCertificate');
        if (percentage >= 85) {
            getCertificateBtn.style.display = 'inline-flex';
        } else {
            getCertificateBtn.style.display = 'none';
        }
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
    
    showCertificateForm() {
        this.showScreen('certificateScreen');
        // Clear form
        document.getElementById('certificateName').value = '';
        document.getElementById('certificateGithub').value = '';
        document.getElementById('generateCertificate').disabled = true;
        
        // Add input validation
        const nameInput = document.getElementById('certificateName');
        const githubInput = document.getElementById('certificateGithub');
        
        const validateForm = () => {
            const hasName = nameInput.value.trim().length > 0;
            const hasGithub = githubInput.value.trim().length > 0;
            document.getElementById('generateCertificate').disabled = !(hasName && hasGithub);
        };
        
        nameInput.addEventListener('input', validateForm);
        githubInput.addEventListener('input', validateForm);
    }
    
    generateCertificate() {
        const name = document.getElementById('certificateName').value.trim();
        const github = document.getElementById('certificateGithub').value.trim();
        
        if (!name || !github) {
            alert('Please fill in all fields');
            return;
        }
        
        // Create certificate canvas
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size (standard certificate dimensions)
        canvas.width = 1200;
        canvas.height = 900;
        
        // Create gradient background
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
        gradient.addColorStop(0, '#1e3a8a');
        gradient.addColorStop(0.5, '#3b82f6');
        gradient.addColorStop(1, '#1e40af');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Add border
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 12;
        ctx.strokeRect(30, 30, canvas.width - 60, canvas.height - 60);
        
        // Inner border
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 4;
        ctx.strokeRect(50, 50, canvas.width - 100, canvas.height - 100);
        
        // Certificate title
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 48px Arial, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('CERTIFICATE OF ACHIEVEMENT', canvas.width / 2, 150);
        
        // Subtitle
        ctx.font = '24px Arial, sans-serif';
        ctx.fillStyle = '#fbbf24';
        ctx.fillText('Production ML Engineering Excellence', canvas.width / 2, 190);
        
        // Main text
        ctx.fillStyle = '#ffffff';
        ctx.font = '32px Arial, sans-serif';
        ctx.fillText('This certifies that', canvas.width / 2, 280);
        
        // Name
        ctx.font = 'bold 42px Arial, sans-serif';
        ctx.fillStyle = '#fbbf24';
        ctx.fillText(name, canvas.width / 2, 340);
        
        // Achievement text
        ctx.fillStyle = '#ffffff';
        ctx.font = '28px Arial, sans-serif';
        ctx.fillText('has successfully demonstrated mastery of', canvas.width / 2, 400);
        ctx.fillText('Production ML Engineering concepts by achieving', canvas.width / 2, 440);
        
        // Score
        const percentage = Math.round((this.score / this.randomizedQuestions.length) * 100);
        ctx.font = 'bold 36px Arial, sans-serif';
        ctx.fillStyle = '#10b981';
        ctx.fillText(`${percentage}% (${this.score}/${this.randomizedQuestions.length}) on the comprehensive quiz`, canvas.width / 2, 490);
        
        // Skills covered
        ctx.fillStyle = '#ffffff';
        ctx.font = '22px Arial, sans-serif';
        ctx.fillText('Covering: Distributed Training • PEFT Techniques • Model Optimization • Architecture Design', canvas.width / 2, 550);
        
        // GitHub handle
        ctx.fillStyle = '#fbbf24';
        ctx.font = 'bold 24px Arial, sans-serif';
        ctx.fillText(`GitHub: @${github}`, canvas.width / 2, 600);
        
        // Date
        const date = new Date().toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '20px Arial, sans-serif';
        ctx.fillText(`Completed on ${date}`, canvas.width / 2, 660);
        
        // Footer
        ctx.fillStyle = '#94a3b8';
        ctx.font = '18px Arial, sans-serif';
        ctx.fillText('Production ML Engineering Handbook', canvas.width / 2, 720);
        ctx.fillText('github.com/tuanthi/distributed-llm-guide', canvas.width / 2, 750);
        
        // Add decorative elements
        this.addCertificateDecorations(ctx, canvas.width, canvas.height);
        
        // Display certificate
        this.displayCertificate(canvas, name, github, percentage);
    }
    
    addCertificateDecorations(ctx, width, height) {
        // Add some decorative stars
        ctx.fillStyle = '#fbbf24';
        const starPositions = [
            [200, 120], [1000, 120], [150, 300], [1050, 300],
            [180, 500], [1020, 500], [220, 700], [980, 700]
        ];
        
        starPositions.forEach(([x, y]) => {
            this.drawStar(ctx, x, y, 15);
        });
        
        // Add ML icons (simplified representations)
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 3;
        
        // Neural network representation
        const nodePositions = [
            [120, 400], [120, 450], [120, 500],
            [1080, 400], [1080, 450], [1080, 500]
        ];
        
        nodePositions.forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, 2 * Math.PI);
            ctx.stroke();
        });
    }
    
    drawStar(ctx, x, y, size) {
        ctx.save();
        ctx.translate(x, y);
        ctx.beginPath();
        
        for (let i = 0; i < 5; i++) {
            ctx.lineTo(Math.cos((18 + i * 72) * Math.PI / 180) * size, 
                      -Math.sin((18 + i * 72) * Math.PI / 180) * size);
            ctx.lineTo(Math.cos((54 + i * 72) * Math.PI / 180) * size * 0.5, 
                      -Math.sin((54 + i * 72) * Math.PI / 180) * size * 0.5);
        }
        
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }
    
    displayCertificate(canvas, name, github, percentage) {
        const certificateContainer = document.getElementById('certificateDisplay');
        certificateContainer.innerHTML = '';
        
        // Add canvas to display
        canvas.style.maxWidth = '100%';
        canvas.style.height = 'auto';
        canvas.style.border = '2px solid #e2e8f0';
        canvas.style.borderRadius = '8px';
        canvas.style.boxShadow = '0 10px 15px -3px rgb(0 0 0 / 0.1)';
        
        certificateContainer.appendChild(canvas);
        
        // Add action buttons
        const actionContainer = document.createElement('div');
        actionContainer.className = 'certificate-actions';
        actionContainer.innerHTML = `
            <button id="downloadCertificate" class="btn btn-primary">
                <i class="fas fa-download"></i> Download Certificate
            </button>
            <button id="shareCertificate" class="btn btn-secondary">
                <i class="fas fa-share"></i> Share on Social Media
            </button>
            <button id="copyShareLink" class="btn btn-outline">
                <i class="fas fa-link"></i> Copy Share Link
            </button>
        `;
        
        certificateContainer.appendChild(actionContainer);
        
        // Bind action events
        document.getElementById('downloadCertificate').addEventListener('click', () => {
            this.downloadCertificate(canvas, name);
        });
        
        document.getElementById('shareCertificate').addEventListener('click', () => {
            this.shareCertificate(name, github, percentage);
        });
        
        document.getElementById('copyShareLink').addEventListener('click', () => {
            this.copyShareLink(name, github, percentage);
        });
        
        // Show certificate display
        this.showScreen('certificateDisplayScreen');
    }
    
    downloadCertificate(canvas, name) {
        // Convert canvas to blob and download
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Production-ML-Engineering-Certificate-${name.replace(/\s+/g, '-')}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 'image/png');
    }
    
    shareCertificate(name, github, percentage) {
        const text = `🎉 I just earned a Production ML Engineering Certificate with ${percentage}% on the comprehensive quiz! 

Skills validated:
✅ Distributed Training (DDP, FSDP)
✅ PEFT Techniques (LoRA, QLoRA)
✅ Model Optimization
✅ MLOps Architecture

Check out the handbook: https://tuanthi.github.io/distributed-llm-guide/
Take the quiz: https://tuanthi.github.io/distributed-llm-guide/quiz/

#MachineLearning #MLOps #AI #ProductionML #DistributedTraining`;
        
        if (navigator.share) {
            navigator.share({
                title: 'Production ML Engineering Certificate',
                text: text,
                url: 'https://tuanthi.github.io/distributed-llm-guide/quiz/'
            });
        } else {
            // Fallback - copy to clipboard
            navigator.clipboard.writeText(text).then(() => {
                alert('Share text copied to clipboard! You can paste it on your social media.');
            });
        }
    }
    
    copyShareLink(name, github, percentage) {
        const url = `https://tuanthi.github.io/distributed-llm-guide/quiz/?shared=true&score=${percentage}&name=${encodeURIComponent(name)}&github=${encodeURIComponent(github)}`;
        
        navigator.clipboard.writeText(url).then(() => {
            alert('Share link copied to clipboard!');
        }).catch(() => {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = url;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            alert('Share link copied to clipboard!');
        });
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