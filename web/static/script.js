const MAX_PRESSURE = 10.0;
const API_URL = '/api/data';

const pressureValueEl = document.getElementById('pressure-value');
const gaugePathEl = document.getElementById('gauge-path');
const alertBoxEl = document.getElementById('alert-box');
const connectionDotEl = document.getElementById('connection-dot');
const connectionStatusEl = document.getElementById('connection-status');

// Setup gauge SVG path length for animation
const gaugePathLength = gaugePathEl.getTotalLength();
gaugePathEl.style.strokeDasharray = gaugePathLength;
gaugePathEl.style.strokeDashoffset = gaugePathLength;

function updateGauge(value) {
    if (value === null || isNaN(value)) {
        pressureValueEl.textContent = '--';
        gaugePathEl.style.strokeDashoffset = gaugePathLength;
        return;
    }

    // Update text
    pressureValueEl.textContent = value.toFixed(1);

    // Calculate percentage (assuming max physical gauge is around 20 PSI or 60 PSI)
    // Adjust max scale dynamically based on value or use a fixed 20 for visual
    const maxScale = Math.max(20, Math.ceil(value / 10) * 10);
    const percentage = Math.min(Math.max(value / maxScale, 0), 1);
    
    // Update SVG arc
    const offset = gaugePathLength - (percentage * gaugePathLength);
    gaugePathEl.style.strokeDashoffset = offset;

    // Handle Colors and Alerts
    if (value > MAX_PRESSURE) {
        gaugePathEl.style.stroke = 'var(--danger)';
        pressureValueEl.style.color = 'var(--danger)';
        alertBoxEl.classList.add('active');
    } else {
        gaugePathEl.style.stroke = 'var(--success)';
        pressureValueEl.style.color = 'var(--text-main)';
        alertBoxEl.classList.remove('active');
    }
}

function setConnectionStatus(connected) {
    if (connected) {
        connectionDotEl.className = 'status-dot online';
        connectionStatusEl.textContent = 'Conectado a Jetson';
    } else {
        connectionDotEl.className = 'status-dot offline';
        connectionStatusEl.textContent = 'Desconectado';
        updateGauge(null);
    }
}

async function fetchPressureData() {
    try {
        const response = await fetch(API_URL);
        if (!response.ok) throw new Error('Network response was not ok');
        
        const data = await response.json();
        setConnectionStatus(true);
        updateGauge(data.pressure);
    } catch (error) {
        console.error('Failed to fetch pressure data:', error);
        setConnectionStatus(false);
    }
}

// Poll every 1 second
setInterval(fetchPressureData, 1000);
fetchPressureData();
