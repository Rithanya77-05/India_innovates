/**
 * ============================================
 * Traffic Management Dashboard — Frontend Logic
 * ============================================
 * v3.0 — 4 Lanes | Real-Time Only | Emergency Corridor
 *
 * Features:
 *  - WebSocket real-time updates (no fake data)
 *  - Start/Stop detection controls
 *  - 4-lane signal + density display
 *  - Full Emergency Green Corridor mode
 *  - GPS trigger + manual deactivate
 */

// ============================================================
// WebSocket Connection
// ============================================================

let ws = null;
let reconnectTimer = null;
let densityHistory = [];
const MAX_HISTORY = 80;
const LANES = ['lane_1', 'lane_2', 'lane_3', 'lane_4'];

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('✅ WebSocket connected');
        updateConnectionStatus(true);
        if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.status === 'waiting') {
                handleWaitingState(data);
            } else {
                handleLiveData(data);
            }
        } catch (e) {
            console.error('Parse error:', e);
        }
    };

    ws.onclose = () => {
        console.log('❌ WebSocket disconnected');
        updateConnectionStatus(false);
        if (!reconnectTimer) {
            reconnectTimer = setTimeout(connectWebSocket, 3000);
        }
    };

    ws.onerror = (e) => console.error('WebSocket error:', e);
}

function updateConnectionStatus(connected) {
    const dot = document.querySelector('.status-dot');
    const txt = document.getElementById('statusText');
    if (connected) {
        dot?.classList.remove('disconnected');
        if (txt) txt.textContent = 'Connected';
    } else {
        dot?.classList.add('disconnected');
        if (txt) txt.textContent = 'Disconnected';
    }
}

// ============================================================
// State Handling
// ============================================================

let isDetectionRunning = false;

function handleWaitingState(data) {
    // Show the waiting overlay if not already running
    if (!isDetectionRunning) {
        showWaitingOverlay();
    }
}

function handleLiveData(data) {
    // If we receive live data, hide the overlay
    hideWaitingOverlay();
    updateDashboard(data);
}

// ============================================================
// Overlay Controls
// ============================================================

function showWaitingOverlay() {
    const overlay = document.getElementById('waitingOverlay');
    overlay?.classList.remove('hidden');
    document.getElementById('liveControls').style.display = 'none';
}

function hideWaitingOverlay() {
    const overlay = document.getElementById('waitingOverlay');
    overlay?.classList.add('hidden');
    document.getElementById('liveControls').style.display = 'flex';
}

async function startDetection() {
    const source = document.getElementById('sourceInput')?.value?.trim() || '0';
    const model = document.getElementById('modelInput')?.value?.trim() ||
        'E:/Traffic_Management/models/best.pt';
    const errEl = document.getElementById('startError');
    const btn = document.getElementById('btnStart');

    errEl.style.display = 'none';
    btn.disabled = true;
    btn.textContent = '⏳ Starting...';

    try {
        // Use query params — matching FastAPI query-param endpoint
        const params = new URLSearchParams({ source, model });
        const resp = await fetch(`/api/start?${params}`, {
            method: 'POST',
        });
        const result = await resp.json();

        if (!resp.ok) {
            const msg = result.error || result.detail || JSON.stringify(result);
            errEl.textContent = `❌ ${msg}`;
            errEl.style.display = 'block';
            btn.disabled = false;
            btn.textContent = '🚀 Start Detection';
            return;
        }

        isDetectionRunning = true;
        document.getElementById('liveSourceLabel').textContent =
            `Source: ${source} | Model: ${result.model || model}`;
        hideWaitingOverlay();
        console.log('✅ Detection started:', result);

    } catch (e) {
        errEl.textContent = `❌ Cannot reach server: ${e.message}`;
        errEl.style.display = 'block';
        btn.disabled = false;
        btn.textContent = '🚀 Start Detection';
    }
}

async function stopDetection() {
    try {
        await fetch('/api/stop', { method: 'POST' });
        isDetectionRunning = false;
        densityHistory = [];
        showWaitingOverlay();
        document.getElementById('btnStart').disabled = false;
        document.getElementById('btnStart').textContent = '🚀 Start Detection';
        resetDashboard();
    } catch (e) {
        console.error('Stop failed:', e);
    }
}

function resetDashboard() {
    LANES.forEach(lane => {
        const valEl = document.getElementById(`density-val-${lane}`);
        const barEl = document.getElementById(`density-bar-${lane}`);
        const timerEl = document.getElementById(`timer-${lane}`);
        if (valEl) valEl.textContent = '—';
        if (barEl) barEl.style.width = '0%';
        if (timerEl) timerEl.textContent = '—';
    });
    ['totalVehicles', 'densityScore'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = '—';
    });
    const cl = document.getElementById('congestionLevel');
    if (cl) cl.textContent = '—';
    updateCorridorLanes(null); // all grey
    clearChart();
}

// ============================================================
// Dashboard Update
// ============================================================

function updateDashboard(data) {
    updateStats(data);
    updateVehicleCounts(data);
    updateDensityBars(data);
    updateSignals(data);
    updateEmergency(data);
    updateEmergencyPanel(data);
    updateDensityHistory(data);
}

// --- Top Stats ---
function updateStats(data) {
    animateValue('totalVehicles', data.total_vehicles || 0);
    animateValue('densityScore', data.density || 0);

    const cEl = document.getElementById('congestionLevel');
    const level = data.congestion || 'LOW';
    if (cEl) {
        cEl.textContent = level;
        cEl.className = 'stat-value congestion-' + level.toLowerCase();
    }
    const fEl = document.getElementById('frameCount');
    if (fEl) fEl.textContent = (data.frame || 0).toLocaleString();
}

// --- Vehicle Counts ---
function updateVehicleCounts(data) {
    const counts = data.vehicle_counts || {};
    const map = {
        'Vehicle': ['Vehicle', 'car'],
        'Bus': ['Bus', 'bus'],
        'Truck': ['Truck', 'truck'],
        'Motorcycle': ['Motorcycle', 'bike'],
        'Ambulance': ['Ambulance', 'ambulance'],
    };
    Object.entries(map).forEach(([name, keys]) => {
        const el = document.getElementById(`count-${name}`);
        if (!el) return;
        let val = 0;
        for (const k of keys) { if (counts[k] !== undefined) { val = counts[k]; break; } }
        const old = parseInt(el.textContent) || 0;
        if (val !== old) {
            el.textContent = val;
            el.classList.add('value-changed');
            setTimeout(() => el.classList.remove('value-changed'), 400);
        }
    });
}

// --- Lane Density Bars (4 lanes) ---
function updateDensityBars(data) {
    const densities = data.lane_densities || {};
    const maxD = 80;
    LANES.forEach(lane => {
        const bar = document.getElementById(`density-bar-${lane}`);
        const val = document.getElementById(`density-val-${lane}`);
        const density = densities[lane] || 0;
        if (bar && val) {
            bar.style.width = `${Math.min((density / maxD) * 100, 100)}%`;
            val.textContent = Math.round(density);
            bar.classList.remove('moderate', 'high', 'critical');
            if (density >= 50) bar.classList.add('critical');
            else if (density >= 30) bar.classList.add('high');
            else if (density >= 15) bar.classList.add('moderate');
        }
    });
}

// --- Traffic Signals (4 lanes) ---
function updateSignals(data) {
    const signals = data.signals || {};
    const timings = data.signal_timings || {};
    const emergency = data.emergency || {};

    LANES.forEach(lane => {
        const unit = document.getElementById(`signal-${lane}`);
        const timer = document.getElementById(`timer-${lane}`);
        const info = signals[lane] || {};

        if (unit) {
            unit.querySelectorAll('.light').forEach(l => l.classList.remove('active'));
            const color = (info.color || 'RED').toLowerCase();
            unit.querySelector(`.light.${color}`)?.classList.add('active');

            if (emergency.mode === 'EMERGENCY') {
                unit.classList.toggle('emergency-active', lane === emergency.green_lane);
            } else {
                unit.classList.remove('emergency-active');
            }
        }
        if (timer) {
            const t = timings[lane] ?? 0;
            timer.textContent = `${t}s`;
            timer.style.color = t > 0 ? 'var(--accent-green)' : 'var(--accent-red)';
        }
    });

    const modeBadge = document.getElementById('signalMode');
    if (modeBadge) {
        if (emergency.mode === 'EMERGENCY') {
            modeBadge.textContent = '🚨 Emergency Override';
            modeBadge.style.cssText = 'background:rgba(255,23,68,.2);color:var(--emergency-red);border-color:rgba(255,23,68,.3);';
        } else {
            modeBadge.textContent = 'Adaptive';
            modeBadge.style.cssText = '';
        }
    }
}

// --- Emergency Banner ---
function updateEmergency(data) {
    const emergency = data.emergency || {};
    const banner = document.getElementById('emergencyBanner');
    const laneEl = document.getElementById('emergencyLane');
    const modeBadge = document.getElementById('modeBadge');
    const header = document.getElementById('mainHeader');

    if (emergency.mode === 'EMERGENCY') {
        banner?.classList.remove('hidden');
        header?.classList.add('emergency-active');
        if (modeBadge) {
            modeBadge.textContent = '🚑 EMERGENCY';
            modeBadge.classList.add('emergency');
        }
        if (laneEl) laneEl.textContent = `→ ${emergency.green_lane || '?'} 🟢`;
    } else {
        banner?.classList.add('hidden');
        header?.classList.remove('emergency-active');
        if (modeBadge) {
            modeBadge.textContent = 'NORMAL';
            modeBadge.classList.remove('emergency');
        }
    }
}

// --- Emergency Panel with Corridor Visualizer ---
function updateEmergencyPanel(data) {
    const emergency = data.emergency || {};
    const badge = document.getElementById('emgStatusBadge');
    const deactivateBtn = document.getElementById('btnDeactivate');
    const timeLabel = document.getElementById('emgTimeLabel');
    const panel = document.getElementById('emergencyPanel');

    if (emergency.mode === 'EMERGENCY') {
        if (badge) {
            badge.textContent = `🚑 ACTIVE — ${emergency.emergency_class || 'Ambulance'}`;
            badge.classList.add('active');
        }
        panel?.classList.add('emergency-mode-active');
        if (deactivateBtn) deactivateBtn.disabled = false;
        if (timeLabel && emergency.time_active != null) {
            let msg = `Active: ${emergency.time_active}s`;
            if (emergency.cooldown_remaining != null) {
                msg += ` | Returns to NORMAL in ${emergency.cooldown_remaining}s`;
            }
            if (emergency.gps_triggered) msg += ' (GPS)';
            timeLabel.textContent = msg;
        }
        updateCorridorLanes(emergency.green_lane);
    } else {
        if (badge) {
            badge.textContent = 'NORMAL';
            badge.classList.remove('active');
        }
        panel?.classList.remove('emergency-mode-active');
        if (deactivateBtn) deactivateBtn.disabled = true;
        if (timeLabel) timeLabel.textContent = '';
        updateCorridorLanes(null);
    }
}

function updateCorridorLanes(greenLane) {
    LANES.forEach(lane => {
        const el = document.getElementById(`corridor-${lane}`);
        if (!el) return;
        const statusEl = el.querySelector('.cl-status');
        el.classList.remove('lane-green', 'lane-red');
        if (greenLane === null) {
            if (statusEl) statusEl.textContent = '⬜';
        } else if (lane === greenLane) {
            el.classList.add('lane-green');
            if (statusEl) statusEl.textContent = '🟢';
        } else {
            el.classList.add('lane-red');
            if (statusEl) statusEl.textContent = '🔴';
        }
    });
}

// ============================================================
// GPS Trigger & Deactivate
// ============================================================

async function triggerGPS() {
    const lane = document.getElementById('gpsTriggerLane')?.value || 'lane_1';
    try {
        const params = new URLSearchParams({ lane, ambulance_id: `AMB-${Date.now()}` });
        const resp = await fetch(`/api/emergency/trigger?${params}`, {
            method: 'POST',
        });
        const result = await resp.json();
        console.log('🚑 GPS trigger:', result);
        showToast(`🚑 Emergency activated on ${lane.replace('_', ' ').toUpperCase()}`, 'emergency');
    } catch (e) {
        showToast('❌ GPS trigger failed: ' + e.message, 'error');
    }
}

async function deactivateEmergency() {
    try {
        const resp = await fetch('/api/emergency/deactivate', { method: 'POST' });
        const result = await resp.json();
        console.log('✅ Deactivated:', result);
        showToast('✅ Emergency deactivated — Returning to Normal mode', 'success');
    } catch (e) {
        showToast('❌ Deactivate failed: ' + e.message, 'error');
    }
}

// ============================================================
// Toast Notification
// ============================================================

function showToast(message, type = 'info') {
    const existing = document.getElementById('toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.id = 'toast';
    toast.textContent = message;
    const colors = {
        emergency: 'rgba(255,23,68,0.95)',
        success: 'rgba(0,230,138,0.95)',
        error: 'rgba(255,80,80,0.95)',
        info: 'rgba(50,100,200,0.95)',
    };
    toast.style.cssText = `
        position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;
        background:${colors[type] || colors.info};color:#fff;
        padding:.75rem 1.25rem;border-radius:12px;font-weight:600;
        font-size:.88rem;box-shadow:0 4px 24px rgba(0,0,0,.4);
        animation:slideIn .3s ease;max-width:380px;
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

// ============================================================
// Density History Chart
// ============================================================

const chartCanvas = document.getElementById('densityChart');
const ctx = chartCanvas ? chartCanvas.getContext('2d') : null;

function clearChart() {
    densityHistory = [];
    if (ctx && chartCanvas) {
        ctx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
    }
}

function updateDensityHistory(data) {
    densityHistory.push({ time: Date.now(), value: data.density || 0, congestion: data.congestion || 'LOW' });
    if (densityHistory.length > MAX_HISTORY) densityHistory.shift();
    drawChart();
}

function drawChart() {
    if (!ctx || !chartCanvas) return;
    const rect = chartCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    chartCanvas.width = rect.width * dpr;
    chartCanvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width, H = rect.height;
    const pad = { top: 24, bottom: 32, left: 50, right: 20 };
    const cW = W - pad.left - pad.right, cH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    if (densityHistory.length < 2) {
        ctx.fillStyle = 'rgba(255,255,255,.15)';
        ctx.font = '500 13px "Outfit",sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Waiting for live data...', W / 2, H / 2);
        return;
    }

    const maxVal = Math.max(80, ...densityHistory.map(d => d.value)) * 1.15;
    ctx.strokeStyle = 'rgba(255,255,255,.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = pad.top + (cH / 5) * i;
        ctx.beginPath(); ctx.setLineDash([4, 6]);
        ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255,255,255,.25)';
        ctx.font = '500 11px "JetBrains Mono",monospace';
        ctx.textAlign = 'right';
        ctx.fillText(Math.round(maxVal - (maxVal / 5) * i), pad.left - 10, y + 4);
    }

    const pts = densityHistory.map((d, i) => ({
        x: pad.left + (i / (MAX_HISTORY - 1)) * cW,
        y: pad.top + cH - (d.value / maxVal) * cH,
    }));

    // Fill gradient
    const fillG = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
    fillG.addColorStop(0, 'rgba(0,230,138,.12)');
    fillG.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.beginPath();
    ctx.moveTo(pts[0].x, H - pad.bottom);
    ctx.lineTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
        const cpx = (pts[i - 1].x + pts[i].x) / 2;
        ctx.bezierCurveTo(cpx, pts[i - 1].y, cpx, pts[i].y, pts[i].x, pts[i].y);
    }
    ctx.lineTo(pts[pts.length - 1].x, H - pad.bottom);
    ctx.closePath(); ctx.fillStyle = fillG; ctx.fill();

    // Line
    const lineG = ctx.createLinearGradient(pad.left, 0, W - pad.right, 0);
    lineG.addColorStop(0, 'rgba(0,230,138,.7)');
    lineG.addColorStop(0.5, 'rgba(0,212,255,.9)');
    lineG.addColorStop(1, 'rgba(77,166,255,1)');
    ctx.strokeStyle = lineG; ctx.lineWidth = 2.5; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
        const cpx = (pts[i - 1].x + pts[i].x) / 2;
        ctx.bezierCurveTo(cpx, pts[i - 1].y, cpx, pts[i].y, pts[i].x, pts[i].y);
    }
    ctx.stroke();

    // End dot
    const last = pts[pts.length - 1];
    ctx.beginPath(); ctx.arc(last.x, last.y, 10, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(77,166,255,.15)'; ctx.fill();
    ctx.beginPath(); ctx.arc(last.x, last.y, 5, 0, Math.PI * 2);
    const dg = ctx.createRadialGradient(last.x, last.y, 0, last.x, last.y, 5);
    dg.addColorStop(0, '#00d4ff'); dg.addColorStop(1, '#4da6ff');
    ctx.fillStyle = dg; ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,.6)'; ctx.lineWidth = 1.5; ctx.stroke();

    const lastD = densityHistory[densityHistory.length - 1];
    ctx.fillStyle = 'rgba(255,255,255,.7)';
    ctx.font = '600 11px "JetBrains Mono",monospace';
    ctx.textAlign = 'center';
    const ly = last.y - 14;
    ctx.fillText(Math.round(lastD.value), last.x, ly < pad.top + 10 ? last.y + 18 : ly);
}

// ============================================================
// Animate Number
// ============================================================

function animateValue(id, newVal) {
    const el = document.getElementById(id);
    if (!el) return;
    const cur = parseFloat(el.textContent) || 0;
    const diff = newVal - cur;
    if (Math.abs(diff) < 0.5) {
        el.textContent = newVal % 1 !== 0 ? newVal.toFixed(1) : newVal;
        return;
    }
    let step = 0, steps = 12;
    const iv = setInterval(() => {
        step++;
        const p = 1 - Math.pow(1 - step / steps, 3);
        const v = cur + diff * p;
        el.textContent = newVal % 1 !== 0 ? v.toFixed(1) : Math.round(v);
        if (step >= steps) {
            clearInterval(iv);
            el.textContent = newVal % 1 !== 0 ? newVal.toFixed(1) : newVal;
        }
    }, 25);
}

// ============================================================
// Zoom-In / Entrance Animations
// ============================================================

function animateEntrance() {
    document.querySelectorAll('.stat-card, .card, .vehicle-type').forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'none';
        setTimeout(() => {
            card.style.transition = 'opacity .5s ease, transform .5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 80 + i * 60);
    });
}

// ============================================================
// Init
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚦 Traffic Management Dashboard v3.0 — 4 Lanes — Ready');
    connectWebSocket();
    animateEntrance();

    // Enter key on inputs triggers start
    ['sourceInput', 'modelInput'].forEach(id => {
        document.getElementById(id)?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') startDetection();
        });
    });

    // Responsive chart resize
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(drawChart, 150);
    });
});
