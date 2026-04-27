/* ═══════════════════════════════════════════════════════════════════════════
   AV Deepfake Detector — Client Logic
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    const HISTORY_KEY = 'av_deepfake_history';
    const MAX_HISTORY = 100;

    // ─── Elements ────────────────────────────────────────────────────────────

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const clearBtn = document.getElementById('clear-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const uploadSection = document.getElementById('upload-section');
    const progressSection = document.getElementById('progress-section');
    const progressLabel = document.getElementById('progress-label');
    const progressPct = document.getElementById('progress-pct');
    const progressBar = document.getElementById('progress-bar');
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');
    const errorText = document.getElementById('error-text');
    const resetBtn = document.getElementById('reset-btn');
    const errorResetBtn = document.getElementById('error-reset-btn');

    // Model selector
    const modelSelect = document.getElementById('model-select');
    const modelBadge = document.getElementById('model-badge');

    // History
    const historyList = document.getElementById('history-list');
    const historyEmpty = document.getElementById('history-empty');
    const historyToolbarCount = document.getElementById('history-toolbar-count');
    const exportCsvBtn = document.getElementById('export-csv-btn');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const historyCount = document.getElementById('history-count');

    let selectedFile = null;
    let activeModel = null;

    // ─── Tab Navigation ──────────────────────────────────────────────────────

    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('tab-' + target).classList.add('active');
            if (target === 'history') renderHistory();
        });
    });

    // ─── Model Selector ──────────────────────────────────────────────────────

    async function loadModels() {
        try {
            const res = await fetch('/api/models');
            if (!res.ok) throw new Error('Failed to load models');
            const data = await res.json();

            modelSelect.innerHTML = '';
            if (!data.models || data.models.length === 0) {
                const opt = document.createElement('option');
                opt.textContent = 'No models found';
                opt.disabled = true;
                modelSelect.appendChild(opt);
                return;
            }

            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.name;
                const auc = m.auc ? ` — AUC ${m.auc.toFixed(2)}` : '';
                const ep = m.epoch ? ` ep${m.epoch}` : '';
                opt.textContent = `${m.name}${ep}${auc}`;
                if (m.name === data.active) opt.selected = true;
                modelSelect.appendChild(opt);
            });

            updateModelBadge(data);
            activeModel = data.active;
        } catch {
            modelSelect.innerHTML = '<option value="">Default model</option>';
            modelBadge.textContent = '';
        }
    }

    function updateModelBadge(data) {
        const selected = data.models.find(m => m.name === (modelSelect.value || data.active));
        if (!selected) { modelBadge.textContent = ''; return; }

        if (selected.name === data.active) {
            modelBadge.textContent = 'active';
            modelBadge.className = 'model-badge active';
        } else {
            const device = selected.device || 'cpu';
            modelBadge.textContent = device.toUpperCase();
            modelBadge.className = 'model-badge ' + device.toLowerCase();
        }
    }

    modelSelect.addEventListener('change', async () => {
        const name = modelSelect.value;
        if (!name || name === activeModel) return;
        try {
            const res = await fetch('/api/models/activate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: name }),
            });
            if (res.ok) {
                activeModel = name;
                const data = await res.json();
                updateModelBadge(data);
            }
        } catch {
            // Silently fall back — server may not support hot-swap
        }
    });

    loadModels();

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
        uploadSection.classList.add('hidden');
        progressSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('video', selectedFile);
        if (modelSelect.value) formData.append('model', modelSelect.value);

        try {
            const result = await uploadWithProgress(formData);
            setProgress('Analyzing with model...', 80);
            await delay(300);
            setProgress('Done!', 100);
            await delay(400);

            // Save to history before showing results
            saveToHistory(result);
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

            xhr.timeout = 120000;
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

        const badge = document.getElementById('verdict-badge');
        badge.className = 'verdict-badge ' + (isReal ? 'real' : 'fake');
        document.getElementById('verdict-text').textContent = data.verdict;

        const confPct = Math.round(data.confidence * 100);
        document.getElementById('confidence-value').textContent = confPct + '%';
        document.getElementById('confidence-value').style.color =
            isReal ? 'var(--real-color)' : 'var(--fake-color)';

        const confBar = document.getElementById('confidence-bar');
        confBar.style.background = isReal
            ? 'linear-gradient(90deg, var(--real-color), #4ade80)'
            : 'linear-gradient(90deg, var(--fake-color), #f87171)';
        setTimeout(() => { confBar.style.width = confPct + '%'; }, 100);

        animateScore('audio', data.scores.audio);
        animateScore('video', data.scores.video);
        animateScore('joint', data.scores.joint);

        document.getElementById('interpretation-text').textContent = data.interpretation;
        document.getElementById('meta-file').textContent = data.file;
        document.getElementById('meta-time').textContent = data.processing_time_sec + 's';
        document.getElementById('meta-windows').textContent = data.windows_analyzed + ' windows';
        document.getElementById('meta-model').textContent = data.model_used || activeModel || '';
    }

    function animateScore(id, value) {
        const bar = document.getElementById(id + '-bar');
        const label = document.getElementById(id + '-value');
        const pct = Math.round(value * 100);

        const color = value >= 0.5
            ? `hsl(${Math.round(142 * value)}, 70%, 50%)`
            : `hsl(${Math.round(30 * value)}, 85%, 55%)`;

        bar.style.background = color;
        label.textContent = value.toFixed(2);
        label.style.color = color;
        setTimeout(() => { bar.style.width = pct + '%'; }, 150);
    }

    // ─── History Storage ─────────────────────────────────────────────────────

    function loadHistory() {
        try {
            return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        } catch {
            return [];
        }
    }

    function saveHistory(items) {
        try {
            localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
        } catch {
            // Storage full or unavailable — silently skip
        }
    }

    function saveToHistory(result) {
        const items = loadHistory();
        const entry = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            file: result.file,
            file_size_mb: result.file_size_mb,
            verdict: result.verdict,
            confidence: result.confidence,
            scores: result.scores,
            processing_time_sec: result.processing_time_sec,
            windows_analyzed: result.windows_analyzed,
            model_used: result.model_used || activeModel || modelSelect.value || 'default',
        };

        items.unshift(entry);
        if (items.length > MAX_HISTORY) items.splice(MAX_HISTORY);
        saveHistory(items);
        updateHistoryBadge(items.length);
    }

    function deleteHistoryItem(id) {
        const items = loadHistory().filter(i => i.id !== id);
        saveHistory(items);
        renderHistory();
        updateHistoryBadge(items.length);
    }

    function updateHistoryBadge(count) {
        if (count > 0) {
            historyCount.textContent = count;
            historyCount.style.display = '';
        } else {
            historyCount.style.display = 'none';
        }
    }

    // ─── History Rendering ────────────────────────────────────────────────────

    function renderHistory() {
        const items = loadHistory();
        updateHistoryBadge(items.length);
        historyToolbarCount.textContent = items.length + ' ' + (items.length === 1 ? 'analysis' : 'analyses');

        if (items.length === 0) {
            historyEmpty.style.display = '';
            historyList.innerHTML = '';
            return;
        }

        historyEmpty.style.display = 'none';
        historyList.innerHTML = '';

        items.forEach((item, idx) => {
            const el = buildHistoryItem(item, idx);
            historyList.appendChild(el);
        });
    }

    function buildHistoryItem(item, idx) {
        const isReal = item.verdict === 'REAL';
        const cls = isReal ? 'real' : 'fake';
        const confPct = Math.round(item.confidence * 100) + '%';
        const date = formatDate(item.timestamp);
        const scores = item.scores || {};

        const div = document.createElement('div');
        div.className = `history-item ${cls}`;
        div.style.animationDelay = (idx * 40) + 'ms';

        div.innerHTML = `
            <div class="history-item-main">
                <div class="history-item-file" title="${escHtml(item.file)}">${escHtml(item.file)}</div>
                <div class="history-item-meta">
                    <span>${date}</span>
                    ${item.processing_time_sec ? `<span>${item.processing_time_sec}s</span>` : ''}
                    ${item.windows_analyzed ? `<span>${item.windows_analyzed} windows</span>` : ''}
                    ${item.model_used ? `<span>${escHtml(item.model_used)}</span>` : ''}
                </div>
                <div class="history-scores-mini" style="margin-top:6px">
                    <span class="history-score-chip">🔊 <span style="color:${scoreColor(scores.audio)}">${fmtScore(scores.audio)}</span></span>
                    <span class="history-score-chip">🎬 <span style="color:${scoreColor(scores.video)}">${fmtScore(scores.video)}</span></span>
                    <span class="history-score-chip">🎯 <span style="color:${scoreColor(scores.joint)}">${fmtScore(scores.joint)}</span></span>
                </div>
            </div>
            <div class="history-item-right">
                <span class="history-verdict-pill ${cls}">${item.verdict}</span>
                <span class="history-confidence">${confPct}</span>
                <button class="history-item-delete" title="Remove entry" aria-label="Delete">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
        `;

        div.querySelector('.history-item-delete').addEventListener('click', (e) => {
            e.stopPropagation();
            div.style.opacity = '0';
            div.style.transform = 'translateX(20px)';
            div.style.transition = 'opacity 0.2s, transform 0.2s';
            setTimeout(() => deleteHistoryItem(item.id), 200);
        });

        return div;
    }

    function scoreColor(v) {
        if (v == null) return 'var(--text-muted)';
        return v >= 0.5
            ? `hsl(${Math.round(142 * v)}, 65%, 55%)`
            : `hsl(${Math.round(30 * v)}, 80%, 60%)`;
    }

    function fmtScore(v) {
        return v != null ? v.toFixed(2) : '—';
    }

    function formatDate(iso) {
        try {
            const d = new Date(iso);
            return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
                + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
        } catch {
            return iso;
        }
    }

    function escHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // ─── CSV Export ──────────────────────────────────────────────────────────

    exportCsvBtn.addEventListener('click', () => {
        const items = loadHistory();
        if (items.length === 0) return;

        const cols = [
            'timestamp', 'file', 'verdict', 'confidence',
            'score_audio', 'score_video', 'score_joint',
            'processing_time_sec', 'windows_analyzed', 'model_used', 'file_size_mb'
        ];

        const rows = items.map(i => [
            i.timestamp,
            `"${(i.file || '').replace(/"/g, '""')}"`,
            i.verdict,
            (i.confidence || 0).toFixed(4),
            (i.scores?.audio || 0).toFixed(4),
            (i.scores?.video || 0).toFixed(4),
            (i.scores?.joint || 0).toFixed(4),
            i.processing_time_sec || '',
            i.windows_analyzed || '',
            `"${(i.model_used || '').replace(/"/g, '""')}"`,
            i.file_size_mb || '',
        ].join(','));

        const csv = [cols.join(','), ...rows].join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `deepfake_history_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    });

    // ─── Clear History ────────────────────────────────────────────────────────

    clearHistoryBtn.addEventListener('click', () => {
        if (!confirm('Delete all history? This cannot be undone.')) return;
        saveHistory([]);
        renderHistory();
        updateHistoryBadge(0);
    });

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

        ['audio', 'video', 'joint'].forEach(id => {
            document.getElementById(id + '-bar').style.width = '0%';
        });
        document.getElementById('confidence-bar').style.width = '0%';
    }

    resetBtn.addEventListener('click', resetAll);
    errorResetBtn.addEventListener('click', resetAll);

    // ─── Init ─────────────────────────────────────────────────────────────────

    updateHistoryBadge(loadHistory().length);

    // ─── Utility ─────────────────────────────────────────────────────────────

    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

})();