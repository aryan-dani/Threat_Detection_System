/* ========================================================
   Threat Detection Console - Minimalist Client Logic
   ======================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const digitalClock = document.getElementById('digitalClock');
    const fpsDisplay = document.getElementById('fpsDisplay');
    const globalStatusBadge = document.getElementById('globalStatusBadge');
    const systemStatusDot = document.getElementById('systemStatusDot');
    const detectedCountOverlay = document.getElementById('detectedCountOverlay');
    const alarmBanner = document.getElementById('alarmBanner');
    const cameraSelect = document.getElementById('cameraSelect');
    const engineSelect = document.getElementById('engineSelect');
    const confSlider = document.getElementById('confSlider');
    const confValue = document.getElementById('confValue');
    const inferFpsDisplay = document.getElementById('inferFpsDisplay');
    const classTagsContainer = document.getElementById('classTagsContainer');
    const toggleAudioBtn = document.getElementById('toggleAudioBtn');
    const snapshotBtn = document.getElementById('snapshotBtn');
    const restartFeedBtn = document.getElementById('restartFeedBtn');
    const logBody = document.getElementById('logBody');
    const clearLogBtn = document.getElementById('clearLogBtn');
    const evidenceGrid = document.getElementById('evidenceGrid');
    const evidenceCount = document.getElementById('evidenceCount');
    const alertAudio = document.getElementById('alertAudio');
    const videoStream = document.getElementById('videoStream');

    let audioEnabled = false;
    let loggedDetections = new Set();
    let evidenceSnapshots = [];
    let lastAudioTime = 0;

    // 1. Digital Clock
    function updateClock() {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        digitalClock.textContent = `${hours}:${minutes}:${seconds} UTC+05:30`;
    }
    setInterval(updateClock, 1000);
    updateClock();

    // 2. Audio Alert Toggle
    toggleAudioBtn.addEventListener('click', () => {
        audioEnabled = !audioEnabled;
        toggleAudioBtn.replaceChildren();
        const iconSpan = document.createElement('span');
        iconSpan.className = 'btn-icon';
        if (audioEnabled) {
            iconSpan.textContent = '🔊';
            toggleAudioBtn.appendChild(iconSpan);
            toggleAudioBtn.appendChild(document.createTextNode(' ALARM: ON'));
            toggleAudioBtn.classList.add('btn-primary');
            toggleAudioBtn.classList.remove('btn-secondary');
            alertAudio.play().then(() => alertAudio.pause()).catch(() => {});
        } else {
            iconSpan.textContent = '🔇';
            toggleAudioBtn.appendChild(iconSpan);
            toggleAudioBtn.appendChild(document.createTextNode(' ALARM: OFF'));
            toggleAudioBtn.classList.remove('btn-primary');
            toggleAudioBtn.classList.add('btn-secondary');
        }
    });

    // 3. Confidence Slider
    confSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value).toFixed(2);
        confValue.textContent = val;
        updateConfig({ confidence_threshold: parseFloat(val) });
    });

    // 4. Camera & Engine Selection
    cameraSelect.addEventListener('change', (e) => {
        updateConfig({ camera_index: parseInt(e.target.value) });
        restartFeed();
    });

    if (engineSelect) {
        engineSelect.addEventListener('change', (e) => {
            updateConfig({ engine: e.target.value });
            updateClassTags(e.target.value);
        });
    }

    function updateClassTags(engine) {
        if (!classTagsContainer) return;
        classTagsContainer.replaceChildren();
        let tags = [];
        if (engine === 'sota_threat' || engine === 'specialized') {
            tags = [
                { text: 'Gun / Pistol' },
                { text: 'Grenade / Bomb' },
                { text: 'Knife / Blade' },
                { text: 'Explosive' },
                { text: 'Immune: Scissors, Bare Arm, Tools', opacity: '0.6' }
            ];
        } else if (engine === 'yolo11_tactical') {
            tags = [
                { text: 'Gun / Firearm' },
                { text: 'Knife / Blade' },
                { text: 'YOLOv11 Tactical Medium', opacity: '0.6' }
            ];
        } else if (engine === 'extended') {
            tags = [
                { text: 'Gun / Pistol' },
                { text: 'Rifle / Firearm' },
                { text: 'Heavy Weapon' },
                { text: 'Knife / Blade' }
            ];
        } else if (engine === 'legacy') {
            tags = [
                { text: 'Pistol / Handgun' },
                { text: 'Combat Knife' },
                { text: 'Legacy Dual-Class Model', opacity: '0.6' }
            ];
        } else {
            tags = [
                { text: 'Handgun / Pistol' },
                { text: 'Rifle / Firearm' },
                { text: 'Knife / Blade' },
                { text: 'Bomb / Explosive' },
                { text: 'Grenade' },
                { text: 'Grounded: Scissors, Tools, Arm', opacity: '0.6' }
            ];
        }

        tags.forEach(t => {
            const span = document.createElement('span');
            span.className = 'tag';
            span.textContent = t.text;
            if (t.opacity) {
                span.style.opacity = t.opacity;
            }
            classTagsContainer.appendChild(span);
        });
    }

    // 5. Restart Feed Button
    restartFeedBtn.addEventListener('click', () => {
        restartFeed();
    });

    function restartFeed() {
        videoStream.src = '';
        setTimeout(() => {
            videoStream.src = `/video_feed?t=${Date.now()}`;
        }, 300);
    }

    videoStream.addEventListener('error', () => {
        setTimeout(restartFeed, 1500);
    });

    // 6. Manual Snapshot
    snapshotBtn.addEventListener('click', async () => {
        try {
            const resp = await fetch('/api/snapshot', { method: 'POST' });
            const data = await resp.json();
            if (data.status === 'success') {
                addEvidenceSnapshot(data.snapshot);
            }
        } catch (err) {
            console.error('Snapshot failed:', err);
        }
    });

    // 7. Clear Log
    clearLogBtn.addEventListener('click', () => {
        logBody.replaceChildren();
        const emptyTr = document.createElement('tr');
        emptyTr.className = 'empty-row';
        const emptyTd = document.createElement('td');
        emptyTd.colSpan = 4;
        emptyTd.textContent = 'No security incidents detected.';
        emptyTr.appendChild(emptyTd);
        logBody.appendChild(emptyTr);
        loggedDetections.clear();
    });

    // Helper: Update Backend Config
    async function updateConfig(config) {
        try {
            await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
        } catch (err) {
            console.error('Config update error:', err);
        }
    }

    // 8. Real-time Status Polling
    async function pollStatus() {
        try {
            const resp = await fetch('/api/status');
            if (!resp.ok) return;
            const data = await resp.json();

            // FPS Displays
            fpsDisplay.textContent = (data.fps || '--') + ' FPS';
            if (inferFpsDisplay) {
                inferFpsDisplay.textContent = (data.inference_fps || '--') + ' FPS';
            }

            // Active Threat Status
            const hasThreat = data.active_threats && data.active_threats.length > 0;
            if (hasThreat) {
                globalStatusBadge.textContent = 'THREAT DETECTED';
                globalStatusBadge.className = 'status-indicator-badge status-alert';
                alarmBanner.classList.add('active');
                systemStatusDot.style.backgroundColor = 'var(--status-threat)';
                detectedCountOverlay.textContent = `THREATS: ${data.active_threats.length}`;

                // Play alarm sound if enabled (throttle every 3s)
                const now = Date.now();
                if (audioEnabled && now - lastAudioTime > 3000) {
                    lastAudioTime = now;
                    alertAudio.currentTime = 0;
                    alertAudio.play().catch(() => {});
                }

                // Add to incident log
                data.active_threats.forEach(t => {
                    const dedupeKey = `${t.label}_${Math.floor(Date.now() / 2000)}`;
                    if (!loggedDetections.has(dedupeKey)) {
                        loggedDetections.add(dedupeKey);
                        addLogEntry(t, dedupeKey);
                    }
                });

                // Periodically prune stale deduplication keys older than 60 seconds
                const currentWindow = Math.floor(Date.now() / 2000);
                for (const key of loggedDetections) {
                    const lastUnderscore = key.lastIndexOf('_');
                    if (lastUnderscore !== -1) {
                        const win = parseInt(key.substring(lastUnderscore + 1), 10);
                        if (!isNaN(win) && currentWindow - win > 30) {
                            loggedDetections.delete(key);
                        }
                    }
                }
            } else {
                globalStatusBadge.textContent = 'MONITORING';
                globalStatusBadge.className = 'status-indicator-badge status-idle';
                alarmBanner.classList.remove('active');
                systemStatusDot.style.backgroundColor = '#ffffff';
                detectedCountOverlay.textContent = 'THREATS: 0';
            }

            // Sync evidence snapshots if new
            if (data.latest_snapshot && (!evidenceSnapshots.length || evidenceSnapshots[0].url !== data.latest_snapshot.url)) {
                addEvidenceSnapshot(data.latest_snapshot);
            }
        } catch (err) {
            // Keep silent on connection retry
        } finally {
            setTimeout(pollStatus, 500);
        }
    }

    // Helper: Add Incident Log Entry
    function addLogEntry(threat, dedupeKey) {
        const emptyRow = logBody.querySelector('.empty-row');
        if (emptyRow) {
            emptyRow.remove();
        }

        const now = new Date();
        const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

        const isCritical = ['bomb', 'grenade', 'explosive', 'rifle', 'firearm', 'gun', 'pistol'].some(c => 
            threat.label.toLowerCase().includes(c)
        );

        const badgeClass = isCritical ? 'threat-high' : 'threat-elevated';
        const levelText = isCritical ? 'CRITICAL' : 'ELEVATED';

        const tr = document.createElement('tr');
        if (dedupeKey) {
            tr.dataset.dedupeKey = dedupeKey;
        }

        const tdTime = document.createElement('td');
        tdTime.textContent = timeStr;

        const tdLabel = document.createElement('td');
        const strongLabel = document.createElement('strong');
        strongLabel.textContent = (threat.label || '').toUpperCase();
        tdLabel.appendChild(strongLabel);

        const tdConf = document.createElement('td');
        const confNum = Number(threat.confidence) || 0;
        tdConf.textContent = `${(confNum * 100).toFixed(0)}%`;

        const tdBadge = document.createElement('td');
        const badgeSpan = document.createElement('span');
        badgeSpan.className = `threat-badge ${badgeClass}`;
        badgeSpan.textContent = levelText;
        tdBadge.appendChild(badgeSpan);

        tr.appendChild(tdTime);
        tr.appendChild(tdLabel);
        tr.appendChild(tdConf);
        tr.appendChild(tdBadge);

        logBody.insertBefore(tr, logBody.firstChild);

        // Limit log entries to 30 in UI and prune backing collection
        while (logBody.children.length > 30) {
            const removed = logBody.lastChild;
            if (removed && removed.dataset && removed.dataset.dedupeKey) {
                loggedDetections.delete(removed.dataset.dedupeKey);
            }
            logBody.removeChild(removed);
        }
    }

    // Helper: Add Evidence Snapshot
    function addEvidenceSnapshot(snapshot) {
        if (!snapshot || !snapshot.url) return;
        evidenceSnapshots.unshift(snapshot);

        const empty = evidenceGrid.querySelector('.evidence-empty');
        if (empty) empty.remove();

        const item = document.createElement('div');
        item.className = 'evidence-item';

        const img = document.createElement('img');
        img.src = snapshot.url;
        img.alt = 'Threat evidence';

        const caption = document.createElement('div');
        caption.className = 'evidence-caption';
        caption.textContent = `${snapshot.timestamp || ''} // ${snapshot.label || 'THREAT'}`;

        item.appendChild(img);
        item.appendChild(caption);

        item.addEventListener('click', () => {
            window.open(snapshot.url, '_blank');
        });

        evidenceGrid.insertBefore(item, evidenceGrid.firstChild);

        // Limit to 10 snapshots in UI and prune backing collection
        while (evidenceGrid.children.length > 10) {
            evidenceGrid.removeChild(evidenceGrid.lastChild);
            evidenceSnapshots.pop();
        }
        while (evidenceSnapshots.length > 10) {
            evidenceSnapshots.pop();
        }
        evidenceCount.textContent = `${evidenceSnapshots.length} ITEMS`;
    }

    // Start Polling
    pollStatus();
});
