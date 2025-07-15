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
        this.SAVE_KEY = 'mlQuizProgress';
        this.autoSaveInterval = null;
        
        this.initializeApp();
    }
    
    initializeApp() {
        this.bindEvents();
        
        // Check for saved progress
        if (this.loadProgress()) {
            const resumeQuiz = confirm('You have a quiz in progress. Would you like to resume where you left off?');
            if (resumeQuiz) {
                this.showScreen('quizScreen');
                this.displayQuestion();
                this.updateProgress();
                // Start auto-save
                this.autoSaveInterval = setInterval(() => this.saveProgress(), 30000); // Save every 30 seconds
                return;
            } else {
                this.clearProgress();
            }
        }
        
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
        
        // Question jump dropdown
        document.getElementById('questionJump').addEventListener('change', (e) => {
            const targetQuestion = parseInt(e.target.value);
            if (!isNaN(targetQuestion) && targetQuestion >= 0 && targetQuestion < this.randomizedQuestions.length) {
                this.currentQuestion = targetQuestion;
                this.displayQuestion();
                this.updateProgress();
            }
        });
    }
    
    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }
    
    saveProgress() {
        const progress = {
            currentQuestion: this.currentQuestion,
            userAnswers: this.userAnswers,
            score: this.score,
            startTime: this.startTime,
            randomizedQuestions: this.randomizedQuestions.map(q => q.id),
            hasShownFeedback: this.hasShownFeedback,
            isAnswered: this.isAnswered
        };
        localStorage.setItem(this.SAVE_KEY, JSON.stringify(progress));
    }
    
    loadProgress() {
        const saved = localStorage.getItem(this.SAVE_KEY);
        if (saved) {
            try {
                const progress = JSON.parse(saved);
                // Reconstruct randomized questions from saved IDs
                const questionMap = {};
                quizData.forEach(q => questionMap[q.id] = q);
                this.randomizedQuestions = progress.randomizedQuestions.map(id => questionMap[id]);
                
                this.currentQuestion = progress.currentQuestion;
                this.userAnswers = progress.userAnswers;
                this.score = progress.score;
                this.startTime = new Date(progress.startTime);
                this.hasShownFeedback = progress.hasShownFeedback;
                this.isAnswered = progress.isAnswered;
                
                return true;
            } catch (e) {
                console.error('Failed to load progress:', e);
                localStorage.removeItem(this.SAVE_KEY);
            }
        }
        return false;
    }
    
    clearProgress() {
        localStorage.removeItem(this.SAVE_KEY);
        if (this.autoSaveInterval) {
            clearInterval(this.autoSaveInterval);
            this.autoSaveInterval = null;
        }
    }
    
    startQuiz() {
        this.clearProgress(); // Clear any existing progress
        this.startTime = new Date();
        this.currentQuestion = 0;
        
        // Randomize questions
        this.randomizedQuestions = this.shuffleArray([...quizData]);
        this.userAnswers = new Array(this.randomizedQuestions.length).fill(null);
        this.score = 0;
        this.isAnswered = false;
        this.hasShownFeedback = false;
        
        // Start auto-save
        this.autoSaveInterval = setInterval(() => this.saveProgress(), 30000); // Save every 30 seconds
        
        // Populate question jump dropdown
        this.populateQuestionJumpDropdown();
        
        this.showScreen('quizScreen');
        this.displayQuestion();
        this.updateProgress();
    }
    
    populateQuestionJumpDropdown() {
        const dropdown = document.getElementById('questionJump');
        dropdown.innerHTML = '<option value="">Jump to...</option>';
        
        // Group questions by category
        const categories = {};
        this.randomizedQuestions.forEach((q, index) => {
            if (!categories[q.category]) {
                categories[q.category] = [];
            }
            categories[q.category].push({ question: q, index: index });
        });
        
        // Add options grouped by category
        Object.keys(categories).forEach(category => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = category;
            
            categories[category].forEach(({ question, index }) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `Q${index + 1}: ${question.question.substring(0, 50)}...`;
                optgroup.appendChild(option);
            });
            
            dropdown.appendChild(optgroup);
        });
    }
    
    displayQuestion() {
        const question = this.randomizedQuestions[this.currentQuestion];
        this.isAnswered = false;
        this.hasShownFeedback = false;
        
        // Update question header
        document.getElementById('questionNumber').textContent = `Question ${this.currentQuestion + 1}`;
        document.getElementById('questionType').textContent = 
            question.type === 'multiple-choice' ? 'Multiple Choice' : 'Short Answer';
        
        // Update dropdown selection
        document.getElementById('questionJump').value = this.currentQuestion;
        
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
        this.saveProgress(); // Save progress after showing feedback
        this.updateQuestionDropdown(); // Update dropdown to show answered status
    }
    
    updateQuestionDropdown() {
        const dropdown = document.getElementById('questionJump');
        const options = dropdown.querySelectorAll('option[value]');
        
        options.forEach(option => {
            const index = parseInt(option.value);
            if (!isNaN(index) && this.userAnswers[index] !== null) {
                option.classList.add('answered');
                // Update text to show answered status
                const baseText = option.textContent.replace(' ✓', '');
                option.textContent = baseText + ' ✓';
            }
        });
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
        this.clearProgress(); // Clear saved progress when quiz is finished
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
        
        // Set canvas size (standard certificate dimensions - A4 landscape ratio)
        canvas.width = 1400;
        canvas.height = 990;
        
        // Create elegant paper background
        const backgroundGradient = ctx.createRadialGradient(canvas.width/2, canvas.height/2, 0, canvas.width/2, canvas.height/2, Math.max(canvas.width, canvas.height)/2);
        backgroundGradient.addColorStop(0, '#fefefe');
        backgroundGradient.addColorStop(0.7, '#f8f9fa');
        backgroundGradient.addColorStop(1, '#e9ecef');
        
        ctx.fillStyle = backgroundGradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Add subtle texture pattern
        this.addPaperTexture(ctx, canvas.width, canvas.height);
        
        // Ornate border design
        this.drawOrnamentBorder(ctx, canvas.width, canvas.height);
        
        // Institution header with crest
        this.drawInstitutionHeader(ctx, canvas.width);
        
        // Certificate title
        ctx.fillStyle = '#1a365d';
        ctx.font = 'bold 52px "Times New Roman", serif';
        ctx.textAlign = 'center';
        ctx.fillText('CERTIFICATE OF EXCELLENCE', canvas.width / 2, 200);
        
        // Decorative line under title
        this.drawDecorativeLine(ctx, canvas.width / 2 - 200, 220, 400);
        
        // Presented to
        ctx.fillStyle = '#2d3748';
        ctx.font = 'italic 28px "Times New Roman", serif';
        ctx.fillText('This is to certify that', canvas.width / 2, 280);
        
        // Recipient name with underline
        ctx.fillStyle = '#1a365d';
        ctx.font = 'bold 48px "Times New Roman", serif';
        const nameMetrics = ctx.measureText(name);
        ctx.fillText(name, canvas.width / 2, 340);
        
        // Elegant underline under name
        ctx.strokeStyle = '#c6ab47';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(canvas.width / 2 - nameMetrics.width / 2 - 20, 355);
        ctx.lineTo(canvas.width / 2 + nameMetrics.width / 2 + 20, 355);
        ctx.stroke();
        
        // Achievement description
        ctx.fillStyle = '#2d3748';
        ctx.font = '26px "Times New Roman", serif';
        ctx.fillText('has successfully demonstrated exceptional proficiency in', canvas.width / 2, 400);
        
        // Subject matter
        ctx.fillStyle = '#1a365d';
        ctx.font = 'bold 32px "Times New Roman", serif';
        ctx.fillText('DISTRIBUTED LLM TRAINING & SERVING IN PRODUCTION', canvas.width / 2, 450);
        
        // Score achievement
        const percentage = Math.round((this.score / this.randomizedQuestions.length) * 100);
        ctx.fillStyle = '#2d3748';
        ctx.font = '24px "Times New Roman", serif';
        ctx.fillText('by achieving an outstanding score of', canvas.width / 2, 490);
        
        // Score highlight
        ctx.fillStyle = '#c6ab47';
        ctx.font = 'bold 36px "Times New Roman", serif';
        ctx.fillText(`${percentage}% (${this.score}/${this.randomizedQuestions.length})`, canvas.width / 2, 530);
        
        // Competencies
        ctx.fillStyle = '#2d3748';
        ctx.font = '20px "Times New Roman", serif';
        ctx.fillText('on comprehensive assessment covering:', canvas.width / 2, 570);
        
        // Skills list with bullet points
        const skills = [
            '• Multi-GPU & Multi-Node LLM Training (DDP, FSDP, DeepSpeed)',
            '• Parameter-Efficient Fine-Tuning (LoRA, QLoRA, AdaLoRA)', 
            '• LLM Optimization & Quantization Techniques',
            '• High-Performance LLM Serving & Deployment'
        ];
        
        ctx.fillStyle = '#1a365d';
        ctx.font = '22px "Times New Roman", serif';
        skills.forEach((skill, index) => {
            ctx.fillText(skill, canvas.width / 2, 610 + (index * 30));
        });
        
        // GitHub credential
        ctx.fillStyle = '#6b7280';
        ctx.font = 'italic 20px "Times New Roman", serif';
        ctx.fillText(`Professional Portfolio: github.com/${github}`, canvas.width / 2, 730);
        
        // Date and signatures
        this.drawSignatureSection(ctx, canvas.width, canvas.height);
        
        // Institution footer
        this.drawInstitutionFooter(ctx, canvas.width, canvas.height);
        
        // Add institutional seal
        this.drawInstitutionalSeal(ctx, canvas.width, canvas.height);
        
        // Display certificate
        this.displayCertificate(canvas, name, github, percentage);
    }
    
    addPaperTexture(ctx, width, height) {
        // Add subtle paper texture
        ctx.globalAlpha = 0.02;
        for (let i = 0; i < 1000; i++) {
            ctx.fillStyle = Math.random() > 0.5 ? '#000000' : '#ffffff';
            ctx.fillRect(Math.random() * width, Math.random() * height, 1, 1);
        }
        ctx.globalAlpha = 1;
    }
    
    drawOrnamentBorder(ctx, width, height) {
        const margin = 25;
        const borderWidth = 15;
        
        // Outer border - elegant dark navy
        ctx.strokeStyle = '#1a365d';
        ctx.lineWidth = borderWidth;
        ctx.strokeRect(margin, margin, width - 2 * margin, height - 2 * margin);
        
        // Inner border - gold accent
        ctx.strokeStyle = '#c6ab47';
        ctx.lineWidth = 4;
        ctx.strokeRect(margin + 20, margin + 20, width - 2 * (margin + 20), height - 2 * (margin + 20));
        
        // Corner ornaments
        this.drawCornerOrnament(ctx, margin + 30, margin + 30, 50);
        this.drawCornerOrnament(ctx, width - margin - 80, margin + 30, 50);
        this.drawCornerOrnament(ctx, margin + 30, height - margin - 80, 50);
        this.drawCornerOrnament(ctx, width - margin - 80, height - margin - 80, 50);
    }
    
    drawCornerOrnament(ctx, x, y, size) {
        ctx.save();
        ctx.translate(x, y);
        ctx.strokeStyle = '#c6ab47';
        ctx.lineWidth = 2;
        
        // Draw ornate corner design
        ctx.beginPath();
        ctx.moveTo(0, size);
        ctx.quadraticCurveTo(size/3, size/3, size, 0);
        ctx.moveTo(0, size/2);
        ctx.quadraticCurveTo(size/4, size/4, size/2, 0);
        ctx.stroke();
        
        // Add small decorative circles
        ctx.fillStyle = '#c6ab47';
        ctx.beginPath();
        ctx.arc(size/4, size/4, 3, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.restore();
    }
    
    drawInstitutionHeader(ctx, width) {
        // Institution name
        ctx.fillStyle = '#1a365d';
        ctx.font = 'bold 28px "Times New Roman", serif';
        ctx.textAlign = 'center';
        ctx.fillText('ADVANCED MACHINE LEARNING FOR PRODUCTION', width / 2, 80);
        
        // Subtitle
        ctx.fillStyle = '#6b7280';
        ctx.font = 'italic 18px "Times New Roman", serif';
        ctx.fillText('Excellence in Production AI Engineering', width / 2, 105);
        
        // Decorative emblems
        this.drawAcademicEmblem(ctx, width / 2 - 250, 85);
        this.drawAcademicEmblem(ctx, width / 2 + 250, 85);
    }
    
    drawAcademicEmblem(ctx, x, y) {
        ctx.save();
        ctx.translate(x, y);
        
        // Shield outline
        ctx.strokeStyle = '#c6ab47';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, -20);
        ctx.lineTo(-15, -10);
        ctx.lineTo(-15, 10);
        ctx.lineTo(0, 20);
        ctx.lineTo(15, 10);
        ctx.lineTo(15, -10);
        ctx.closePath();
        ctx.stroke();
        
        // Inner design
        ctx.fillStyle = '#1a365d';
        ctx.beginPath();
        ctx.arc(0, 0, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // Small decorative lines
        ctx.strokeStyle = '#c6ab47';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(-8, -5);
        ctx.lineTo(8, -5);
        ctx.moveTo(-8, 5);
        ctx.lineTo(8, 5);
        ctx.stroke();
        
        ctx.restore();
    }
    
    drawDecorativeLine(ctx, x, y, width) {
        const centerX = x + width / 2;
        
        // Main line
        ctx.strokeStyle = '#c6ab47';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + width, y);
        ctx.stroke();
        
        // Center ornament
        ctx.fillStyle = '#c6ab47';
        ctx.beginPath();
        ctx.arc(centerX, y, 6, 0, 2 * Math.PI);
        ctx.fill();
        
        // Side flourishes
        this.drawFlourish(ctx, x - 20, y);
        this.drawFlourish(ctx, x + width + 20, y);
    }
    
    drawFlourish(ctx, x, y) {
        ctx.strokeStyle = '#c6ab47';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x - 10, y);
        ctx.quadraticCurveTo(x, y - 8, x + 10, y);
        ctx.quadraticCurveTo(x, y + 8, x - 10, y);
        ctx.stroke();
    }
    
    drawSignatureSection(ctx, width, height) {
        const y = height - 180;
        const date = new Date().toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
        
        // Date
        ctx.fillStyle = '#2d3748';
        ctx.font = '18px "Times New Roman", serif';
        ctx.textAlign = 'left';
        ctx.fillText('Date of Achievement:', 150, y);
        ctx.font = 'bold 18px "Times New Roman", serif';
        ctx.fillText(date, 150, y + 25);
        
        // Signature lines and titles
        const sigY = y + 70;
        
        // Authority signature
        ctx.textAlign = 'center';
        ctx.font = '16px "Times New Roman", serif';
        ctx.fillStyle = '#6b7280';
        
        // Signature line
        ctx.strokeStyle = '#2d3748';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(width - 400, sigY);
        ctx.lineTo(width - 200, sigY);
        ctx.stroke();
        
        ctx.fillText('Assessment Authority', width - 300, sigY + 20);
        ctx.fillText('Advanced ML for Production', width - 300, sigY + 40);
        
        // Authentication number
        ctx.textAlign = 'left';
        ctx.font = '14px "Times New Roman", serif';
        ctx.fillStyle = '#9ca3af';
        const certNumber = `AMLP-${Date.now().toString(36).toUpperCase()}`;
        ctx.fillText(`Certificate No: ${certNumber}`, 150, sigY + 40);
    }
    
    drawInstitutionFooter(ctx, width, height) {
        ctx.fillStyle = '#6b7280';
        ctx.font = '14px "Times New Roman", serif';
        ctx.textAlign = 'center';
        
        ctx.fillText('This certificate validates demonstrated competency in advanced machine learning engineering', width / 2, height - 80);
        ctx.fillText('as assessed by the Advanced Machine Learning for Production curriculum standards', width / 2, height - 60);
        
        // Website and quiz link
        ctx.fillStyle = '#1a365d';
        ctx.font = 'bold 16px "Times New Roman", serif';
        ctx.fillText('Take the Assessment: tuanthi.github.io/distributed-llm-guide/quiz', width / 2, height - 45);
        
        ctx.fillStyle = '#9ca3af';
        ctx.font = '12px "Times New Roman", serif';
        ctx.fillText('github.com/tuanthi/distributed-llm-guide • Accredited Assessment Program', width / 2, height - 25);
    }
    
    drawInstitutionalSeal(ctx, width, height) {
        const sealX = width - 150;
        const sealY = height - 250;
        const radius = 60;
        
        ctx.save();
        ctx.translate(sealX, sealY);
        
        // Outer circle
        ctx.strokeStyle = '#c6ab47';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.arc(0, 0, radius, 0, 2 * Math.PI);
        ctx.stroke();
        
        // Inner circle
        ctx.strokeStyle = '#1a365d';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(0, 0, radius - 15, 0, 2 * Math.PI);
        ctx.stroke();
        
        // Center emblem
        ctx.fillStyle = '#1a365d';
        ctx.beginPath();
        ctx.arc(0, 0, 20, 0, 2 * Math.PI);
        ctx.fill();
        
        // ML symbol in center
        ctx.strokeStyle = '#c6ab47';
        ctx.lineWidth = 3;
        ctx.font = 'bold 16px "Times New Roman", serif';
        ctx.fillStyle = '#c6ab47';
        ctx.textAlign = 'center';
        ctx.fillText('ML', 0, 6);
        
        // Radiating lines
        ctx.strokeStyle = '#c6ab47';
        ctx.lineWidth = 2;
        for (let i = 0; i < 8; i++) {
            const angle = (i * Math.PI) / 4;
            ctx.beginPath();
            ctx.moveTo(Math.cos(angle) * 30, Math.sin(angle) * 30);
            ctx.lineTo(Math.cos(angle) * 45, Math.sin(angle) * 45);
            ctx.stroke();
        }
        
        // Seal text
        ctx.fillStyle = '#6b7280';
        ctx.font = 'bold 10px "Times New Roman", serif';
        ctx.textAlign = 'center';
        ctx.fillText('EXCELLENCE', 0, -70);
        ctx.fillText('VERIFIED', 0, 80);
        
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
        const text = `🎓 Proud to announce: I've earned an official Certificate of Excellence in Distributed LLM Training & Serving in Production!

🏆 Achievement: ${percentage}% score on comprehensive LLM engineering assessment by Advanced Machine Learning for Production

📋 Verified competencies:
✅ Multi-GPU & Multi-Node LLM Training (DDP, FSDP, DeepSpeed)
✅ Parameter-Efficient Fine-Tuning (LoRA, QLoRA, AdaLoRA)
✅ LLM Optimization & Quantization Techniques
✅ High-Performance LLM Serving & Deployment

This accredited certification validates my expertise in distributed LLM training and production deployment at scale.

🔗 Handbook: https://tuanthi.github.io/distributed-llm-guide/
🧠 Take the assessment: https://tuanthi.github.io/distributed-llm-guide/quiz/

#LLM #DistributedTraining #LLMOps #AI #LargeLanguageModels #DeepLearning #FSDP #LoRA #Certification #LLMEngineering`;
        
        if (navigator.share) {
            navigator.share({
                title: 'Certificate: Distributed LLM Training & Serving in Production',
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