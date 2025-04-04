// static/js/script.js

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('stressForm');
    
    form.addEventListener('submit', async function(event) {
        event.preventDefault();
        const submitButton = form.querySelector('button[type="submit"]');
        submitButton.disabled = true;
    
        try {
            const formData = new FormData();  // Use FormData for file upload
            formData.append('student_id', document.getElementById('student_id').value);
            formData.append('text', document.getElementById('text').value);
            formData.append('image', document.getElementById('image').files[0]);  // File object
            formData.append('study_behavior', JSON.stringify([
                parseInt(document.getElementById('study_time').value) || 0,
                parseInt(document.getElementById('social_media').value) || 0,
                parseInt(document.getElementById('sleep_hours').value) || 0,
                parseInt(document.getElementById('deadlines').value) || 0
            ]));
    
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData  // Send FormData instead of JSON
            });
    
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
    
            const result = await response.json();
            console.log('Received result:', result);
            displayResult(result);
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing request: ' + error.message);
        } finally {
            submitButton.disabled = false;
        }
    });
});

function getStressClass(score) {
    if (score <= 0.25) return 'normal';
    if (score <= 0.50) return 'moderate';
    if (score <= 0.75) return 'high';
    return 'severe';
}

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    
    // Check for emergency conditions
    const isEmergency = result.stress_score > 0.65 || result.depression_score > 0.65;
    
    // Update gauges and scores first
    const stressGauge = document.getElementById('stress-gauge');
    const depressionGauge = document.getElementById('depression-gauge');
    
    stressGauge.style.width = `${result.stress_score * 100}%`;
    depressionGauge.style.width = `${result.depression_score * 100}%`;
    
    stressGauge.className = `progress ${getStressClass(result.stress_score)}`;
    depressionGauge.className = `progress ${getStressClass(result.depression_score)}`;
    
    document.getElementById('stress-level').textContent = result.stress_level;
    document.getElementById('depression-level').textContent = result.depression_level;
    document.getElementById('stress-score').textContent = result.stress_score.toFixed(2);
    document.getElementById('depression-score').textContent = result.depression_score.toFixed(2);
    
    // Handle alerts (boolean values from backend)
    const counselorMessage = result.alert_counselor === true 
        ? "High risk detected. Recommend consulting a counselor." 
        : null;
    
    // Proctor message with meeting recommendation for high stress/depression
    let proctorMessage = null;
    if (result.alert_proctor === true) {
        if (result.stress_score > 0.65 || result.depression_score > 0.65) {
            proctorMessage = "Due to high stress or depression, a meeting with your proctor is recommended.";
        } else {
            proctorMessage = "Student overwhelmed with workload or poor sleep. Review study schedule.";
        }
    }
    
    // Update result div content with differentiated alerts
    resultDiv.innerHTML = `
        <div class="analysis-result">
            <h2>Analysis Results 📊</h2>
            <div class="stress-info">
                <div class="stress-level ${getStressClass(result.stress_score)}">
                    <h3>Mental Health Status 🧠</h3>
                    <div class="scores-container">
                        <p class="score-display">Stress Level: ${result.stress_level} 📈</p>
                        <p class="score-display">Stress Score: ${result.stress_score.toFixed(2)} 🎯</p>
                        <p class="score-display">Depression Level: ${result.depression_level} 💭</p>
                        <p class="score-display">Depression Score: ${result.depression_score.toFixed(2)} 📉</p>
                    </div>
                </div>
            </div>
            ${counselorMessage ? `
                <div class="alert counselor-alert">
                    <span class="alert-title">Counselor Alert 🚨</span>
                    <p>${counselorMessage}</p>
                </div>
            ` : ''}
            ${proctorMessage ? `
                <div class="alert proctor-alert">
                    <span class="alert-title">Proctor Notification 📋</span>
                    <p>${proctorMessage}</p>
                </div>
            ` : ''}
            ${isEmergency ? `
                <div class="emergency-warning">
                    <p class="warning-text">⚠️ Your results indicate high levels of distress. 
                    Please seek immediate support from available resources.</p>
                </div>
            ` : ''}
            <div class="recommendations">
                <h3>Recommendations 💡</h3>
                <ul>
                    ${result.recommendations ? result.recommendations.map(rec => `<li>💪 ${rec}</li>`).join('') : ''}
                </ul>
            </div>
        </div>
    `;
    
    // Show emergency alert for critical cases
    if (isEmergency) {
        const emergencyAlert = document.createElement('div');
        emergencyAlert.className = 'emergency-alert';
        emergencyAlert.innerHTML = `
            <div class="emergency-content">
                <h2>⚠️ EMERGENCY ALERT ⚠️</h2>
                <p>Your mental health indicators show critical levels.</p>
                <div class="emergency-contacts">
                    <h3>Immediate Support Available:</h3>
                    <ul>
                        <li>🏥 Campus Counseling: 1800-123-4567</li>
                        <li>🆘 24/7 Crisis Helpline: 1800-273-8255</li>
                        <li>👨‍⚕️ Student Health Center: SJT-VIT, Room-801</li>
                    </ul>
                </div>
                <button onclick="this.parentElement.parentElement.style.display='none'" class="close-alert">
                    Acknowledge
                </button>
            </div>
        `;
        document.body.insertBefore(emergencyAlert, document.body.firstChild);
    }
    
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}