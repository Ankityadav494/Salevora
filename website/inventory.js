/* ================================================
   Salevora — inventory.js
   Real-time Inventory Intelligence Logic
   ================================================ */

const API = 'http://localhost:8000';
const WS  = 'ws://localhost:8000/ws/inventory';
const REFRESH_SECS = 30;

let allItems   = [];
let ws         = null;
let countdown  = REFRESH_SECS;
let cdInterval = null;

// ---- Helpers --------------------------------------------------------

function fmtCur(v) {
  if (!v && v !== 0) return '—';
  if (v >= 1_000_000) return '$' + (v/1_000_000).toFixed(1) + 'M';
  if (v >= 1_000)     return '$' + (v/1_000).toFixed(1) + 'K';
  return '$' + v.toFixed(2);
}

function fmtNum(v) {
  return (v||0).toLocaleString('en-US', { maximumFractionDigits: 0 });
}

function statusIcon(s) {
  return { ok:'🟢', warning:'🟡', critical:'🔴', stockout:'⛔' }[s] || '—';
}

function toast(msg, type='success', ms=3500) {
  const c = document.getElementById('toastContainer');
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => { t.style.opacity='0'; t.style.transition='opacity 0.3s'; setTimeout(()=>t.remove(),350); }, ms);
}

function setLiveBadge(connected) {
  const b = document.getElementById('liveBadge');
  if (connected) {
    b.className = 'live-badge';
    b.innerHTML = '<span class="live-dot"></span> LIVE';
  } else {
    b.className = 'live-badge disconnected';
    b.innerHTML = '<span class="live-dot"></span> OFFLINE';
  }
}

// ---- Countdown Ring -------------------------------------------------

function startCountdown() {
  clearInterval(cdInterval);
  countdown = REFRESH_SECS;
  updateRing();
  cdInterval = setInterval(() => {
    countdown--;
    updateRing();
    if (countdown <= 0) {
      countdown = REFRESH_SECS;
      fetchAll();
    }
  }, 1000);
}

function updateRing() {
  const pct = (countdown / REFRESH_SECS) * 100;
  const fill = document.getElementById('ringFill');
  const label = document.getElementById('ringLabel');
  if (fill)  fill.setAttribute('stroke-dasharray', `${pct} 100`);
  if (label) label.textContent = countdown;
}

// ---- API Calls ------------------------------------------------------

async function fetchKPIs() {
  try {
    const r = await fetch(`${API}/api/inventory/kpis`);
    if (!r.ok) throw new Error();
    const d = await r.json();
    const bar = document.getElementById('kpiBar');
    bar.innerHTML = `
      <div class="inv-kpi"><div class="inv-kpi-val">${d.total_skus}</div><div class="inv-kpi-lab">Total SKUs</div></div>
      <div class="inv-kpi"><div class="inv-kpi-val">${fmtCur(d.total_value)}</div><div class="inv-kpi-lab">Stock Value</div></div>
      <div class="inv-kpi critical"><div class="inv-kpi-val">${d.critical}</div><div class="inv-kpi-lab">🔴 Critical</div></div>
      <div class="inv-kpi warning"><div class="inv-kpi-val">${d.at_risk}</div><div class="inv-kpi-lab">🟡 At Risk</div></div>
      <div class="inv-kpi"><div class="inv-kpi-val">${d.avg_days_stock}d</div><div class="inv-kpi-lab">Avg Days Stock</div></div>
      <div class="inv-kpi"><div class="inv-kpi-val">${d.in_stock_pct}%</div><div class="inv-kpi-lab">In-Stock Rate</div></div>`;
    const nt = new Date(d.updated_at);
    document.getElementById('refreshInfo').textContent = `Updated ${nt.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}`;
    document.getElementById('footerTime').textContent = nt.toLocaleString();
  } catch(e) {
    document.getElementById('kpiBar').innerHTML = '<div style="color:var(--red);padding:.5rem">⚠️ Cannot reach API at localhost:8000. Make sure FastAPI is running.</div>';
  }
}

async function fetchLive() {
  try {
    const r = await fetch(`${API}/api/inventory/live`);
    if (!r.ok) throw new Error();
    const d = await r.json();
    allItems = d.items;
    renderTable(allItems);
    populateForecastDropdown(allItems);
    renderEOQ(allItems);
  } catch(e) {}
}

async function fetchAlerts() {
  try {
    const r = await fetch(`${API}/api/inventory/alerts`);
    if (!r.ok) throw new Error();
    const d = await r.json();
    renderAlerts(d.alerts);

    // Alert strip
    const strip = document.getElementById('alertStrip');
    if (d.count > 0) {
      strip.style.display = '';
      const critical = d.alerts.filter(a=>a.status==='critical').length;
      document.getElementById('alertStripText').textContent =
        `⚠️ ${d.count} item${d.count>1?'s':''} need restock${critical?` — ${critical} CRITICAL`:''}. Scroll to Restock Alerts. `;
    } else {
      strip.style.display = 'none';
    }
  } catch(e) {}
}

async function fetchABC() {
  try {
    const r = await fetch(`${API}/api/inventory/abc`);
    if (!r.ok) throw new Error();
    const d = await r.json();
    renderABC(d);
  } catch(e) {}
}

async function fetchAll() {
  await Promise.all([fetchKPIs(), fetchLive(), fetchAlerts(), fetchABC()]);
}

// ---- Table Render ---------------------------------------------------

function renderTable(items) {
  const body = document.getElementById('stockBody');
  if (!items.length) {
    body.innerHTML = '<tr><td colspan="10" class="table-loading">No items match the filter.</td></tr>';
    return;
  }
  body.innerHTML = items.map(i => {
    const pct = Math.min(100, i.stock_pct || 0);
    const daysColor = i.days_of_stock <= 3 ? 'var(--red)' : i.days_of_stock <= 7 ? 'var(--yellow)' : 'var(--green)';
    return `<tr data-status="${i.status}" data-cat="${i.category}" data-name="${i.name.toLowerCase()}" data-sku="${i.sku.toLowerCase()}">
      <td><span style="font-family:'Outfit',sans-serif;font-weight:700;font-size:0.78rem;color:var(--accent)">${i.sku}</span></td>
      <td style="font-weight:600;max-width:160px">${i.name}</td>
      <td style="color:var(--text-2)">${i.category}</td>
      <td><span class="abc-inline ${i.abc_class}">${i.abc_class}</span></td>
      <td>
        <div class="stock-bar-wrap">
          <div class="stock-bar-bg"><div class="stock-bar-fill ${i.status}" style="width:${pct}%"></div></div>
          <span style="font-weight:700;white-space:nowrap">${fmtNum(i.stock)}</span>
        </div>
      </td>
      <td style="color:var(--text-2)">${fmtNum(i.reorder_pt)}</td>
      <td style="font-weight:700;color:${daysColor}">${i.days_of_stock <= 999 ? i.days_of_stock+'d' : '∞'}</td>
      <td style="color:var(--text-2)">${i.daily_demand}/day</td>
      <td><span class="status-badge ${i.status}">${statusIcon(i.status)} ${i.status.charAt(0).toUpperCase()+i.status.slice(1)}</span></td>
      <td><button class="btn-restock" onclick="restock('${i.sku}','${i.name}')">+ Restock</button></td>
    </tr>`;
  }).join('');
}

function filterTable() {
  const q    = document.getElementById('skuSearch').value.toLowerCase();
  const st   = document.getElementById('statusFilter').value;
  const cat  = document.getElementById('catFilter').value;
  const rows = document.querySelectorAll('#stockBody tr[data-sku]');
  rows.forEach(tr => {
    const matchQ   = !q   || tr.dataset.name.includes(q) || tr.dataset.sku.includes(q);
    const matchSt  = !st  || tr.dataset.status === st;
    const matchCat = !cat || tr.dataset.cat === cat;
    tr.style.display = (matchQ && matchSt && matchCat) ? '' : 'none';
  });
}

// ---- Alerts Render --------------------------------------------------

function renderAlerts(alerts) {
  const body = document.getElementById('alertsBody');
  const count = document.getElementById('alertCount');
  count.textContent = alerts.length;

  if (!alerts.length) {
    body.innerHTML = '<div class="alerts-empty">✅ All items fully stocked. No restock needed.</div>';
    return;
  }

  body.innerHTML = alerts.map(a => `
    <div class="alert-card ${a.status}">
      <div class="alert-card-top">
        <div>
          <div class="alert-sku">${a.sku} · ${a.category}</div>
          <div class="alert-name">${a.name}</div>
        </div>
        <div style="text-align:right">
          <div class="alert-days ${a.status}">${a.days_of_stock}</div>
          <div class="alert-days-lab">days left</div>
        </div>
      </div>
      <div class="alert-meta">
        <span>Stock: <strong>${fmtNum(a.stock)}</strong></span>
        <span>Reorder Pt: <strong>${fmtNum(a.reorder_pt)}</strong></span>
        <span>Order Qty: <strong>${fmtNum(a.order_qty)}</strong></span>
        <span class="order-cost">Cost: ${fmtCur(a.order_cost)}</span>
        <span>Lead: <strong>${a.lead_time}d</strong></span>
      </div>
      <div class="alert-actions">
        <button class="btn-order-now ${a.status}" onclick="restock('${a.sku}','${a.name}')">
          📦 Place Restock Order (${fmtNum(a.order_qty)} units)
        </button>
      </div>
    </div>`).join('');
}

// ---- Forecast Chart -------------------------------------------------

function populateForecastDropdown(items) {
  const sel = document.getElementById('forecastSku');
  const cur = sel.value;
  sel.innerHTML = '<option value="">Select SKU…</option>' +
    items.map(i => `<option value="${i.sku}" ${i.sku===cur?'selected':''}>${i.sku} — ${i.name}</option>`).join('');
  if (!cur && items.length) loadForecast(items[0].sku);
}

async function loadForecast(sku) {
  if (!sku) return;
  const sel = document.getElementById('forecastSku');
  sel.value = sku;
  try {
    const r = await fetch(`${API}/api/inventory/forecast?sku=${encodeURIComponent(sku)}`);
    if (!r.ok) throw new Error();
    const d = await r.json();
    const fc = d.forecasts[0];
    if (!fc) return;
    const days    = fc.forecast.map(f=>f.day);
    const stocks  = fc.forecast.map(f=>f.projected_stock);
    const demands = fc.forecast.map(f=>f.demand);
    const plotLayout = {
      paper_bgcolor:'transparent', plot_bgcolor:'transparent',
      font:{ color:'#a5b4d0', family:'Inter,sans-serif', size:11 },
      xaxis:{ gridcolor:'rgba(99,102,241,0.08)', tickfont:{size:10} },
      yaxis:{ gridcolor:'rgba(99,102,241,0.08)', tickprefix:'', tickformat:',d', title:{text:'Units',font:{size:10,color:'#556080'}} },
      margin:{l:45,r:10,t:15,b:55},
      legend:{orientation:'h',y:-0.25,font:{size:10}},
      shapes: fc.reorder_recommended ? [{
        type:'line', x0:0, x1:1, xref:'paper',
        y0:fc.reorder_recommended, y1:fc.reorder_recommended,
        line:{color:'rgba(249,115,22,0.5)', width:1.5, dash:'dot'},
      }] : [],
    };
    Plotly.newPlot('forecastChart', [
      {
        x:days, y:stocks, type:'scatter', mode:'lines+markers', name:'Projected Stock',
        line:{color:'#6366f1',width:2.2,shape:'spline'}, fill:'tozeroy', fillcolor:'rgba(99,102,241,0.08)',
        marker:{size:6,color:'#6366f1',line:{width:1.5,color:'#fff'}},
        hovertemplate:'<b>%{x}</b><br>Stock: %{y:,d} units<extra></extra>',
      },
      {
        x:days, y:demands, type:'bar', name:'Est. Daily Demand',
        marker:{color:'rgba(6,182,212,0.35)'}, yaxis:'y2',
        hovertemplate:'<b>%{x}</b><br>Demand: %{y:.1f}/day<extra></extra>',
      },
    ], {
      ...plotLayout,
      yaxis2:{overlaying:'y', side:'right', showgrid:false, tickformat:'.1f',
               title:{text:'Demand/day',font:{size:10,color:'#556080'}}, range:[0, Math.max(...demands)*3]},
    }, { displayModeBar:false, responsive:true });

    const stockoutDay = fc.stockout_day;
    document.getElementById('forecastMeta').innerHTML = `
      <div class="forecast-meta-item"><div class="forecast-meta-val">${fmtNum(fc.current_stock)}</div><div class="forecast-meta-lab">Current Stock</div></div>
      <div class="forecast-meta-item"><div class="forecast-meta-val">${fc.daily_demand}/day</div><div class="forecast-meta-lab">Avg Daily Demand</div></div>
      <div class="forecast-meta-item"><div class="forecast-meta-val">${fmtNum(fc.reorder_recommended)}</div><div class="forecast-meta-lab">EOQ Recommendation</div></div>
      <div class="forecast-meta-item"><div class="forecast-meta-val" style="color:${stockoutDay?'var(--red)':'var(--green)'}">${stockoutDay || '> 7 days'}</div><div class="forecast-meta-lab">Stockout Projection</div></div>`;
  } catch(e) {
    document.getElementById('forecastChart').innerHTML = '<div style="padding:2rem;color:var(--text-3)">Failed to load forecast data.</div>';
  }
}

// ---- ABC Chart ------------------------------------------------------

function renderABC(data) {
  const classes = ['A','B','C'];
  const colors  = ['rgba(99,102,241,0.8)','rgba(6,182,212,0.7)','rgba(16,185,129,0.65)'];
  const counts  = classes.map(c => data.items.filter(i=>i.abc_class===c).length);
  const values  = classes.map(c => data.items.filter(i=>i.abc_class===c).reduce((s,i)=>s+i.annual_value,0));
  Plotly.newPlot('abcChart', [
    {
      labels: classes.map((c,i) => `${c} — ${counts[i]} SKUs`),
      values: values,
      type: 'pie',
      hole: 0.52,
      marker: { colors },
      textinfo: 'label+percent',
      hovertemplate: '<b>Class %{label}</b><br>Annual Value: $%{value:,.0f}<extra></extra>',
      textfont: { size: 12, color: '#f0f4ff', family:'Inter,sans-serif' },
    }
  ], {
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'transparent',
    font: { color:'#a5b4d0', family:'Inter,sans-serif', size:11 },
    showlegend: false,
    margin: { l:10, r:10, t:15, b:10 },
    annotations: [{
      text: `$${(data.total_annual_value/1000).toFixed(0)}K<br><span style="font-size:9px">Annual</span>`,
      x:0.5, y:0.5, showarrow:false,
      font: { size:16, color:'#f0f4ff', family:'Outfit,sans-serif' },
    }],
  }, { displayModeBar:false, responsive:true });
}

// ---- EOQ Grid -------------------------------------------------------

function renderEOQ(items) {
  const grid = document.getElementById('eoqGrid');
  const sorted = [...items].sort((a,b)=>{
    const scoreA = a.status==='critical'?0:a.status==='warning'?1:2;
    const scoreB = b.status==='critical'?0:b.status==='warning'?1:2;
    return scoreA - scoreB;
  });
  grid.innerHTML = sorted.map(i=>`
    <div class="eoq-card">
      <div class="eoq-sku">${i.sku} <span class="abc-inline ${i.abc_class}">${i.abc_class}</span></div>
      <div class="eoq-name">${i.name}</div>
      <div class="eoq-val">${fmtNum(i.eoq)}</div>
      <div class="eoq-unit">units / order</div>
      <div class="eoq-cost">Order cost: ${fmtCur(i.eoq * i.cost)}</div>
    </div>`).join('');
}

// ---- Restock Action -------------------------------------------------

async function restock(sku, name) {
  try {
    const r = await fetch(`${API}/api/inventory/restock/${encodeURIComponent(sku)}`, { method:'POST' });
    if (!r.ok) throw new Error();
    const d = await r.json();
    toast(`✅ Restocked ${d.qty_added} units of "${name}"`, 'success');
    // Highlight button as restocked
    const btns = document.querySelectorAll(`.btn-restock`);
    btns.forEach(b => { if (b.closest('tr')?.querySelector('td span')?.textContent === sku) b.classList.add('restocked'); });
    await fetchAll();
  } catch(e) {
    toast(`⚠️ Could not reach API. Make sure FastAPI is running on port 8000.`, 'error', 5000);
  }
}

// ---- WebSocket ------------------------------------------------------

function connectWS() {
  try {
    ws = new WebSocket(WS);
    ws.onopen = () => {
      setLiveBadge(true);
      toast('🟢 Live data stream connected!', 'success');
    };
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'snapshot') {
        allItems = data.items;
        renderTable(allItems);
        renderEOQ(allItems);
        // Also refresh alerts + KPIs from REST for accuracy
        fetchKPIs();
        fetchAlerts();
        const nt = new Date(data.updated_at);
        document.getElementById('refreshInfo').textContent = `Live stream · ${nt.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}`;
        document.getElementById('footerTime').textContent = nt.toLocaleString();
      }
    };
    ws.onerror = () => {};
    ws.onclose = () => {
      setLiveBadge(false);
      // Retry in 5s
      setTimeout(connectWS, 5000);
    };
  } catch(e) {
    setLiveBadge(false);
  }
}

// ---- Init -----------------------------------------------------------

window.addEventListener('DOMContentLoaded', async () => {
  // Initial full fetch
  await fetchAll();
  startCountdown();
  // Connect WebSocket for real-time updates
  connectWS();
});
