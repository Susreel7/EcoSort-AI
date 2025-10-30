// Global variables
let stream = null;
let liveStream = null;
let liveIntervalId = null;
let currentTab = 'upload';
const DEBUG = false; // Set to false for production
let lastGeoSentAt = 0;

// Tab switching
document.addEventListener('DOMContentLoaded', function() {
    if (DEBUG) console.log('DOM loaded, initializing app...');
    
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            switchTab(tabId);
        });
    });
    
    // Upload functionality
    setupUpload();
    setupCamera();
    setupBatchUpload();
    setupLive();
    
    if (DEBUG) console.log('App initialization complete!');
    // Stop camera streams on unload to release device
    window.addEventListener('beforeunload', () => {
        try { if (stream) stream.getTracks().forEach(t => t.stop()); } catch(_) {}
        try { if (liveStream) liveStream.getTracks().forEach(t => t.stop()); } catch(_) {}
    });
});

function switchTab(tabId) {
    // Update active tab button
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
    
    // Update active tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabId}-tab`).classList.add('active');
    
    currentTab = tabId;
    // Lazy init analytics when tab becomes active
    if (tabId === 'analytics' && typeof window.initAnalytics === 'function') {
        window.initAnalytics();
    }
    // Auto stop live if leaving live tab
    if (tabId !== 'live' && typeof stopLive === 'function') {
        stopLive();
    }
}

function setupUpload() {
    const uploadBox = document.getElementById('uploadBox');
    const imageInput = document.getElementById('imageInput');
    
    if (!uploadBox || !imageInput) {
        console.error('Upload elements not found!');
        return;
    }
    
    // Drag and drop functionality
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('drag-over');
    });
    
    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('drag-over');
    });
    
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleSingleImage(files[0]);
        }
    });
    
    // Click to upload
    uploadBox.addEventListener('click', (e) => {
        // Don't trigger if clicking on the input itself
        if (e.target !== imageInput) {
            imageInput.click();
        }
    });
    
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSingleImage(e.target.files[0]);
        }
    });
}

function setupBatchUpload() {
    const batchUploadBox = document.getElementById('batchUploadBox');
    const batchImageInput = document.getElementById('batchImageInput');
    
    if (!batchUploadBox || !batchImageInput) {
        console.error('Batch upload elements not found!');
        return;
    }
    
    // Drag and drop functionality
    batchUploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        batchUploadBox.classList.add('drag-over');
    });
    
    batchUploadBox.addEventListener('dragleave', () => {
        batchUploadBox.classList.remove('drag-over');
    });
    
    batchUploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        batchUploadBox.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleBatchImages(files);
        }
    });
    
    // Click to upload
    batchUploadBox.addEventListener('click', (e) => {
        // Don't trigger if clicking on the input itself
        if (e.target !== batchImageInput) {
            batchImageInput.click();
        }
    });
    
    batchImageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleBatchImages(e.target.files);
        }
    });
}

function setupCamera() {
    const startBtn = document.getElementById('startCamera');
    const captureBtn = document.getElementById('captureBtn');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    
    if (!startBtn || !captureBtn || !video || !canvas) {
        console.error('Camera elements not found!');
        return;
    }
    
    startBtn.addEventListener('click', startCamera);
    captureBtn.addEventListener('click', capturePhoto);
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('video');
        video.srcObject = stream;
        
        document.getElementById('startCamera').style.display = 'none';
        document.getElementById('captureBtn').style.display = 'inline-flex';
    } catch (err) {
        alert('Error accessing camera: ' + err.message);
    }
}

function capturePhoto() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    ctx.drawImage(video, 0, 0);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg');
    
    // Send to server immediately
    sendCameraImage(imageData);
}

function resetCamera() {
    // Stop current stream
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    // Reset UI
    const video = document.getElementById('video');
    const startBtn = document.getElementById('startCamera');
    const captureBtn = document.getElementById('captureBtn');
    
    video.style.display = 'block';
    video.srcObject = null;
    startBtn.style.display = 'inline-flex';
    captureBtn.style.display = 'none';
}

function setupLive() {
    const startLiveBtn = document.getElementById('startLiveBtn');
    const stopLiveBtn = document.getElementById('stopLiveBtn');
    const liveVideo = document.getElementById('liveVideo');
    const liveCanvas = document.getElementById('liveCanvas');
    
    if (!startLiveBtn || !stopLiveBtn || !liveVideo || !liveCanvas) {
        return;
    }
    
    startLiveBtn.addEventListener('click', startLive);
    stopLiveBtn.addEventListener('click', stopLive);
}

async function startLive() {
    try {
        if (liveStream) return;
        liveStream = await navigator.mediaDevices.getUserMedia({ video: true });
        const liveVideo = document.getElementById('liveVideo');
        liveVideo.srcObject = liveStream;
        
        document.getElementById('startLiveBtn').style.display = 'none';
        document.getElementById('stopLiveBtn').style.display = 'inline-flex';
        document.getElementById('liveResult').style.display = 'block';
        
        // Send a frame every 800ms (adjust for performance)
        liveIntervalId = setInterval(captureAndSendLiveFrame, 800);
    } catch (err) {
        alert('Error accessing camera: ' + err.message);
    }
}

function stopLive() {
    if (liveIntervalId) {
        clearInterval(liveIntervalId);
        liveIntervalId = null;
    }
    if (liveStream) {
        liveStream.getTracks().forEach(t => t.stop());
        liveStream = null;
    }
    const liveVideo = document.getElementById('liveVideo');
    liveVideo.srcObject = null;
    document.getElementById('startLiveBtn').style.display = 'inline-flex';
    document.getElementById('stopLiveBtn').style.display = 'none';
}

async function captureAndSendLiveFrame() {
    const liveVideo = document.getElementById('liveVideo');
    const liveCanvas = document.getElementById('liveCanvas');
    const ctx = liveCanvas.getContext('2d');
    
    if (!liveVideo.videoWidth || !liveVideo.videoHeight) return;
    
    // Downscale for bandwidth/perf
    const targetWidth = 320;
    const scale = targetWidth / liveVideo.videoWidth;
    liveCanvas.width = targetWidth;
    liveCanvas.height = Math.round(liveVideo.videoHeight * scale);
    
    ctx.drawImage(liveVideo, 0, 0, liveCanvas.width, liveCanvas.height);
    const imageData = liveCanvas.toDataURL('image/jpeg', 0.7);
    
    try {
        const response = await fetch('/predict_live', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const result = await response.json();
        if (result && result.success) {
            displayLiveResult(result);
            // Fire-and-forget log with geolocation if available
            maybeSendGeolog(result);
        }
    } catch (e) {
        // Ignore transient errors during live mode
    }
}

function displayLiveResult(result) {
    document.getElementById('liveResultEmoji').textContent = result.class_info.emoji;
    document.getElementById('liveResultClass').textContent = result.prediction.toUpperCase();
    document.getElementById('liveConfidenceText').textContent = `${(result.confidence * 100).toFixed(2)}% Confidence`;
    document.getElementById('liveCategoryBadge').textContent = result.class_info.category;
    
    const probabilityList = document.getElementById('liveProbabilityList');
    probabilityList.innerHTML = '';
    result.results.forEach(item => {
        const div = document.createElement('div');
        div.className = 'probability-item';
        div.innerHTML = `
            <span class="emoji">${item.emoji}</span>
            <span class="class-name">${item.class.charAt(0).toUpperCase() + item.class.slice(1)}</span>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${item.probability * 100}%"></div>
            </div>
            <span class="percentage">${(item.probability * 100).toFixed(1)}%</span>
        `;
        probabilityList.appendChild(div);
    });

    // Tips
    if (result.class_info && Array.isArray(result.class_info.tips)) {
        ensureTips('live', result.class_info.tips);
    }
}

async function handleSingleImage(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }
    
    showLoading();
    
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            displaySingleResult(result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error uploading image: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function handleBatchImages(files) {
    const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
        alert('Please select image files.');
        return;
    }
    
    showLoading();
    
    const formData = new FormData();
    imageFiles.forEach((file) => {
        formData.append('images', file);
    });
    
    try {
        const response = await fetch('/predict_batch', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            displayBatchResults(result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Batch upload error:', error);
        alert('Error uploading batch images: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function sendCameraImage(imageData) {
    showLoading();
    
    try {
        const response = await fetch('/predict_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayCameraResult(result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        hideLoading();
        // Reset camera for next capture
        resetCamera();
    }
}

function displaySingleResult(result) {
    const resultSection = document.getElementById('uploadResult');
    
    // Update result elements
    document.getElementById('resultEmoji').textContent = result.class_info.emoji;
    document.getElementById('resultClass').textContent = result.prediction.toUpperCase();
    document.getElementById('confidenceText').textContent = `${(result.confidence * 100).toFixed(2)}% Confidence`;
    document.getElementById('categoryBadge').textContent = result.class_info.category;
    
    // Update probability list
    const probabilityList = document.getElementById('probabilityList');
    probabilityList.innerHTML = '';
    
    result.results.forEach(item => {
        const div = document.createElement('div');
        div.className = 'probability-item';
        div.innerHTML = `
            <span class="emoji">${item.emoji}</span>
            <span class="class-name">${item.class.charAt(0).toUpperCase() + item.class.slice(1)}</span>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${item.probability * 100}%"></div>
            </div>
            <span class="percentage">${(item.probability * 100).toFixed(1)}%</span>
        `;
        probabilityList.appendChild(div);
    });
    
    resultSection.style.display = 'block';

    // Tips
    if (result.class_info && Array.isArray(result.class_info.tips)) {
        ensureTips('upload', result.class_info.tips);
    }
}

// Optional geolocation logging
async function maybeSendGeolog(result) {
    try {
        if (!navigator.geolocation) return;
        // Throttle to at most once every 20s
        const now = Date.now();
        if (now - lastGeoSentAt < 20000) return;
        navigator.geolocation.getCurrentPosition(async (pos) => {
            try {
                await fetch('/log_geo', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prediction: result.prediction,
                        confidence: result.confidence,
                        latitude: pos.coords.latitude,
                        longitude: pos.coords.longitude,
                        source: 'live'
                    })
                });
                lastGeoSentAt = Date.now();
            } catch (_e) {}
        });
    } catch (_e) {}
}

function displayCameraResult(result) {
    const resultSection = document.getElementById('cameraResult');
    
    // Update result elements
    document.getElementById('cameraResultEmoji').textContent = result.class_info.emoji;
    document.getElementById('cameraResultClass').textContent = result.prediction.toUpperCase();
    document.getElementById('cameraConfidenceText').textContent = `${(result.confidence * 100).toFixed(2)}% Confidence`;
    document.getElementById('cameraCategoryBadge').textContent = result.class_info.category;
    
    // Update probability list
    const probabilityList = document.getElementById('cameraProbabilityList');
    probabilityList.innerHTML = '';
    
    result.results.forEach(item => {
        const div = document.createElement('div');
        div.className = 'probability-item';
        div.innerHTML = `
            <span class="emoji">${item.emoji}</span>
            <span class="class-name">${item.class.charAt(0).toUpperCase() + item.class.slice(1)}</span>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${item.probability * 100}%"></div>
            </div>
            <span class="percentage">${(item.probability * 100).toFixed(1)}%</span>
        `;
        probabilityList.appendChild(div);
    });
    
    resultSection.style.display = 'block';

    // Tips
    if (result.class_info && Array.isArray(result.class_info.tips)) {
        ensureTips('camera', result.class_info.tips);
    }
}

function ensureTips(scope, tips) {
    let containerId;
    if (scope === 'upload') containerId = 'uploadTips';
    if (scope === 'camera') containerId = 'cameraTips';
    if (scope === 'live') containerId = 'liveTips';
    if (!containerId) return;
    let container = document.getElementById(containerId);
    if (!container) return;
    container.style.display = 'block';
    const ul = container.querySelector('ul');
    ul.innerHTML = '';
    tips.forEach(t => {
        const li = document.createElement('li');
        li.textContent = t;
        ul.appendChild(li);
    });
}

function displayBatchResults(result) {
    const resultsSection = document.getElementById('batchResults');
    
    // Update statistics
    document.getElementById('totalImages').textContent = result.statistics.total_images;
    document.getElementById('avgConfidence').textContent = `${(result.statistics.avg_confidence * 100).toFixed(1)}%`;
    document.getElementById('highConfidence').textContent = result.statistics.high_confidence;
    document.getElementById('successfulCount').textContent = result.statistics.successful;
    
    // Update individual results
    const individualResults = document.getElementById('individualResults');
    individualResults.innerHTML = '';
    
    result.results.forEach(item => {
        const div = document.createElement('div');
        div.className = 'result-item';
        
        const confidenceClass = item.confidence > 0.8 ? 'success' : item.confidence > 0.5 ? 'warning' : 'error';
        
        div.innerHTML = `
            <img src="data:image/png;base64,${item.image_base64}" class="thumbnail" alt="${item.filename}">
            <div class="details">
                <div class="filename">${item.filename}</div>
                <div class="prediction">${item.prediction.charAt(0).toUpperCase() + item.prediction.slice(1)}</div>
                <div class="confidence">${(item.confidence * 100).toFixed(2)}% confidence</div>
            </div>
            <div class="status ${confidenceClass}">
                ${confidenceClass === 'success' ? 'High' : confidenceClass === 'warning' ? 'Moderate' : 'Low'} confidence
            </div>
        `;
        individualResults.appendChild(div);
    });
    
    resultsSection.style.display = 'block';
}

function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

// CSS for drag over effect
const style = document.createElement('style');
style.textContent = `
    .drag-over {
        border-color: #00DD77 !important;
        background: rgba(0, 255, 136, 0.1) !important;
        transform: scale(1.02);
    }
    
    .status.success {
        color: #00FF88;
        font-weight: bold;
    }
    
    .status.warning {
        color: #FFA500;
        font-weight: bold;
    }
    
    .status.error {
        color: #FF6B6B;
        font-weight: bold;
    }
`;
document.head.appendChild(style);
