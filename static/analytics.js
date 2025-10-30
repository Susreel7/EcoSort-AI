async function fetchData() {
    const res = await fetch('/analytics/data');
    const json = await res.json();
    return json.entries || [];
}

function buildCounts(entries) {
    const counts = {};
    entries.forEach(e => {
        const label = e.prediction;
        if (!label) return;
        counts[label] = (counts[label] || 0) + 1;
    });
    return counts;
}

function buildAccuracyOverTime(entries) {
    // Use confidence as proxy for accuracy trend; bucket by hour
    const buckets = {};
    entries.forEach(e => {
        if (typeof e.timestamp !== 'number') return;
        const hour = new Date(e.timestamp * 1000);
        hour.setMinutes(0,0,0);
        const key = hour.toISOString();
        if (!buckets[key]) buckets[key] = [];
        if (typeof e.confidence === 'number') {
            buckets[key].push(e.confidence);
        }
    });
    const labels = Object.keys(buckets).sort();
    const values = labels.map(k => {
        const arr = buckets[k];
        return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
    });
    return { labels, values };
}

function buildLeaderboard(counts) {
    return Object.entries(counts)
        .sort((a,b) => b[1]-a[1])
        .slice(0, 7);
}

function renderCountsChart(ctx, counts) {
    const labels = Object.keys(counts);
    const data = Object.values(counts);
    // eslint-disable-next-line no-undef
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: ['#B07B3F','#FF6B3A','#20C1C1','#D9534F','#9AA2A6','#E6D89C','#4C7BF5']
            }]
        },
        options: { plugins: { legend: { position: 'bottom' } } }
    });
}

function renderAccuracyChart(ctx, series) {
    // eslint-disable-next-line no-undef
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: series.labels,
            datasets: [{
                label: 'Avg confidence',
                data: series.values.map(v => (v*100).toFixed(2)),
                borderColor: '#4C7BF5',
                tension: 0.25
            }]
        },
        options: { scales: { y: { beginAtZero: true, ticks: { callback: v => v + '%' } } } }
    });
}

function renderMap(entries) {
    const mapEl = document.getElementById('map');
    // eslint-disable-next-line no-undef
    const map = L.map(mapEl).setView([20,0], 2);
    // eslint-disable-next-line no-undef
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
        attribution: '&copy; OpenStreetMap'
    }).addTo(map);
    entries.forEach(e => {
        if (typeof e.latitude === 'number' && typeof e.longitude === 'number') {
            // eslint-disable-next-line no-undef
            L.circleMarker([e.latitude, e.longitude], { radius: 5, color: '#4C7BF5' })
              .addTo(map)
              .bindPopup(`${e.prediction} (${Math.round((e.confidence||0)*100)}%)`);
        }
    });
}

function renderLeaderboard(el, top) {
    el.innerHTML = '';
    el.className = 'leaderboard';
    const max = top.length ? top[0][1] : 1;
    // Load CLASS_INFO from embedded JSON (avoid inline script errors)
    if (!window.CLASS_INFO) {
        try {
            const dataTag = document.getElementById('classInfoJson');
            if (dataTag) window.CLASS_INFO = JSON.parse(dataTag.textContent || '{}');
        } catch (_) {}
    }
    top.forEach(([label, count], idx) => {
        const emoji = label && window.CLASS_INFO && window.CLASS_INFO[label] ? window.CLASS_INFO[label].emoji : '';
        const div = document.createElement('div');
        div.className = 'leaderboard-item';
        const percent = Math.max(5, Math.round((count / max) * 100));
        div.innerHTML = `
            <div class="leaderboard-rank">${idx + 1}</div>
            <div class="leaderboard-label"><span class="leaderboard-emoji">${emoji}</span>${label.toUpperCase()}</div>
            <div class="leaderboard-count">${count}</div>
            <div class="leaderboard-bar"><div class="leaderboard-fill" style="width:${percent}%"></div></div>
        `;
        el.appendChild(div);
    });
}

window.initAnalytics = async function initAnalytics() {
    if (!document.getElementById('analytics-tab')?.classList.contains('active')) return;
    try {
        const entries = await fetchData();
        const counts = buildCounts(entries);
        const trend = buildAccuracyOverTime(entries);
        const top = buildLeaderboard(counts);
        
        renderCountsChart(document.getElementById('countsChart'), counts);
        renderAccuracyChart(document.getElementById('accuracyChart'), trend);
        renderMap(entries);
        renderLeaderboard(document.getElementById('leaderboard'), top);
    } catch (e) {
        // eslint-disable-next-line no-console
        console.error('Analytics init error', e);
    }
}


