/* ═══════════════════════════════════════════════════════════════════════════
   AV Deepfake Detector — Client Logic
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // Elements
    const dropZone       = document.getElementById('drop-zone');
    const fileInput      = document.getElementById('file-input');
    const fileInfo       = document.getElementById('file-info');
    const fileName       = document.getElementById('file-name');
    const fileSize       = document.getElementById('file-size');
    const clearBtn       = document.getElementById('clear-btn');
    const analyzeBtn     = document.getElementById('analyze-btn');
    const uploadSection  = document.getElementById('upload-section');
    const progressSection= document.getElementById('progress-section');
    const progressLabel  = document.getElementById('progress-label');
    const progressPct    = document.getElementById('progress-pct');
    const progressBar    = document.getElementById('progress-bar');
    const resultsSection = document.getElementById('results-section');
    const errorSection   = document.getElementById('error-section');
    const errorText      = document.getElementById('error-text');
    const resetBtn       = document.getElementById('reset-btn');
    const errorResetBtn  = document.getElementById('error-reset-btn');

    let selectedFile = null;

    // ─── Drop Zone ───────────────────────────────────────────────────────────

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    // ─── File Handling ───────────────────────────────────────────────────────

    function handleFile(file) {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        const allowed = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'];

        if (!allowed.includes(ext)) {
            showError('Unsupported format. Please use: ' + allowed.join(', '));
            return;
        }

        if (file.size > 500 * 1024 * 1024) {
            showError('File too large. Maximum size is 500MB.');
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatSize(file.size);
        fileInfo.classList.remove('hidden');
        analyzeBtn.classList.remove('hidden');
        dropZone.style.display = 'none';
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
        dropZone.style.display = '';
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // ─── Analysis ────────────────────────────────────────────────────────────

    analyzeBtn.addEventListener('click', () => {
        if (!selectedFile) return;
        startAnalysis();
    });

    async function startAnalysis() {
        // Show progress
        uploadSection.classList.add('hidden');
        progressSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('video', selectedFile);

        try {
            // Upload with progress tracking
            const result = await uploadWithProgress(formData);

            // Show analyzing stage
            setProgress('Analyzing with model...', 80);
            await delay(300);
            setProgress('Done!', 100);
            await delay(400);

            // Display results
            showResults(result);
        } catch (err) {
            showError(err.message || 'An unexpected error occurred.');
        }
    }

    function uploadWithProgress(formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const pct = Math.round((e.loaded / e.total) * 70);
                    setProgress('Uploading video...', pct);
                }
            });

            xhr.addEventListener('load', () => {
                try {
                    const data = JSON.parse(xhr.responseText);
                    if (xhr.status >= 400) {
                        reject(new Error(data.error || 'Server error'));
                    } else {
                        resolve(data);
                    }
                } catch {
                    reject(new Error('Invalid response from server'));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('Network error — is the server running?'));
            });

            xhr.addEventListener('timeout', () => {
                reject(new Error('Request timed out. The video may be too long.'));
            });

            xhr.timeout = 120000; // 2 minutes
            xhr.open('POST', '/api/analyze');
            xhr.send(formData);
        });
    }

    function setProgress(label, pct) {
        progressLabel.textContent = label;
        progressPct.textContent = pct + '%';
        progressBar.style.width = pct + '%';
    }

    // ─── Results Display ─────────────────────────────────────────────────────

    function showResults(data) {
        progressSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        const isReal = data.verdict === 'REAL';

        // Verdict
        const badge = document.getElementById('verdict-badge');
        badge.className = 'verdict-badge ' + (isReal ? 'real' : 'fake');
        document.getElementById('verdict-text').textContent = data.verdict;

        // Confidence
        const confPct = Math.round(data.confidence * 100);
        document.getElementById('confidence-value').textContent = confPct + '%';
        document.getElementById('confidence-value').style.color =
            isReal ? 'var(--real-color)' : 'var(--fake-color)';

        const confBar = document.getElementById('confidence-bar');
        confBar.style.background = isReal
            ? 'linear-gradient(90deg, var(--real-color), #4ade80)'
            : 'linear-gradient(90deg, var(--fake-color), #f87171)';
        setTimeout(() => { confBar.style.width = confPct + '%'; }, 100);

        // Score bars
        animateScore('audio', data.scores.audio);
        animateScore('video', data.scores.video);
        animateScore('joint', data.scores.joint);

        // Interpretation
        document.getElementById('interpretation-text').textContent = data.interpretation;

        // Meta
        document.getElementById('meta-file').textContent = data.file;
        document.getElementById('meta-time').textContent = data.processing_time_sec + 's';
        document.getElementById('meta-windows').textContent = data.windows_analyzed + ' windows';
    }

    function animateScore(id, value) {
        const bar = document.getElementById(id + '-bar');
        const label = document.getElementById(id + '-value');
        const pct = Math.round(value * 100);

        // Color: green if >= 0.5, red if < 0.5
        const color = value >= 0.5
            ? `hsl(${Math.round(142 * value)}, 70%, 50%)`
            : `hsl(${Math.round(30 * value)}, 85%, 55%)`;

        bar.style.background = color;
        label.textContent = value.toFixed(2);
        label.style.color = color;

        setTimeout(() => { bar.style.width = pct + '%'; }, 150);
    }

    // ─── Error ───────────────────────────────────────────────────────────────

    function showError(message) {
        uploadSection.classList.add('hidden');
        progressSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.remove('hidden');
        errorText.textContent = message;
    }

    // ─── Reset ───────────────────────────────────────────────────────────────

    function resetAll() {
        clearFile();
        uploadSection.classList.remove('hidden');
        progressSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
        progressBar.style.width = '0%';

        // Reset score bars
        ['audio', 'video', 'joint'].forEach(id => {
            document.getElementById(id + '-bar').style.width = '0%';
        });
        document.getElementById('confidence-bar').style.width = '0%';
    }

    resetBtn.addEventListener('click', resetAll);
    errorResetBtn.addEventListener('click', resetAll);

    // ─── Utility ─────────────────────────────────────────────────────────────

    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

})();
