/* ================================================
   Salevora — inventory.js
   Real-time Inventory Intelligence Logic
   ================================================ */

const API = (typeof API_BASE !== 'undefined' ? API_BASE : 'http://localhost:8000');
const WS  = API.replace(/^http/, 'ws') + '/ws/inventory';
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
  return { ok:'', warning:'', critical:'', stockout:'' }[s] || '';
}

function abcLabel(c) {
  return { A: 'Top', B: 'Mid', C: 'Slow' }[c] || c;
}

function statusLabel(s) {
  return { critical:'Urgent', warning:'Watch', ok:'Good', stockout:'Out of stock' }[s] || s;
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
    b.innerHTML = '<span class="live-dot"></span> Live';
  } else {
    b.className = 'live-badge disconnected';
    b.innerHTML = '<span class="live-dot"></span> Offline';
  }
}

function populateCategoryFilter(items) {
  const sel = document.getElementById('catFilter');
  if (!sel) return;
  const cats = [...new Set(items.map(i => i.category).filter(Boolean))].sort();
  const cur = sel.value;
  sel.innerHTML = '<option value="">All Categories</option>' +
    cats.map(c => `<option ${c === cur ? 'selected' : ''}>${c}</option>`).join('');
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
    const d = await SalevoraAPI.inventory.kpis();
    const bar = document.getElementById('kpiBar');
    const icons = typeof INV_KPI_ICONS !== 'undefined' ? INV_KPI_ICONS : ['package','dollar','alert','bell','clock','check'];
    const items = [
      { val: d.total_skus, lab: 'Products', cls: '' },
      { val: fmtCur(d.total_value), lab: 'Stock value', cls: '' },
      { val: d.critical, lab: 'Urgent', cls: 'critical' },
      { val: d.forecast_alerts ?? '—', lab: 'May run out', cls: 'warning' },
      { val: `${d.avg_days_stock}d`, lab: 'Days of stock', cls: '' },
      { val: `${d.in_stock_pct}%`, lab: 'In stock', cls: '' },
    ];
    bar.innerHTML = items.map((item, i) => `
      <div class="stock-kpi ${item.cls}">
        <span class="stock-kpi-icon">${typeof svIcon === 'function' ? svIcon(icons[i], 18) : ''}</span>
        <div class="stock-kpi-val">${item.val}</div>
        <div class="stock-kpi-lab">${item.lab}</div>
      </div>`).join('');
    const nt = new Date(d.updated_at);
    document.getElementById('refreshInfo').textContent = `Updated ${nt.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}`;
    document.getElementById('footerTime').textContent = nt.toLocaleString();
  } catch(e) {
    document.getElementById('kpiBar').innerHTML = '<div style="color:var(--red);padding:.5rem;grid-column:1/-1">Cannot connect. Make sure the app is running, then refresh.</div>';
  }
}

async function fetchLive() {
  try {
    const d = await SalevoraAPI.inventory.live();
    allItems = d.items;
    renderTable(allItems);
    populateCategoryFilter(allItems);
    populateForecastDropdown(allItems);
    renderEOQ(allItems);
  } catch(e) {}
}

async function fetchAlerts() {
  try {
    const d = await SalevoraAPI.inventory.alerts();
    renderAlerts(d.alerts);

    // Alert strip
    const strip = document.getElementById('alertStrip');
    if (d.count > 0) {
      strip.style.display = '';
      const critical = d.alerts.filter(a=>a.status==='critical').length;
      document.getElementById('alertStripText').textContent =
        `${d.count} item${d.count>1?'s':''} may run out soon${critical?` — ${critical} urgent`:''}.`;
    } else {
      strip.style.display = 'none';
    }
  } catch(e) {}
}

async function fetchABC() {
  try {
    const d = await SalevoraAPI.inventory.abc();
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
    body.innerHTML = '<tr><td colspan="11" class="table-loading">Nothing here yet — upload your sales file on the dashboard first.</td></tr>';
    return;
  }
  body.innerHTML = items.map(i => {
    const pct = Math.min(100, i.stock_pct || 0);
    const daysColor = i.days_of_stock <= 3 ? 'var(--red)' : i.days_of_stock <= 7 ? 'var(--yellow)' : 'var(--green)';
    const fcColor = i.forecast_shortfall ? 'var(--red)' : 'var(--text-2)';
    return `<tr data-status="${i.status}" data-cat="${i.category}" data-name="${i.name.toLowerCase()}" data-sku="${i.sku.toLowerCase()}">
      <td><span style="font-family:var(--font-display);font-weight:400;font-size:0.82rem;color:var(--accent)">${i.sku}</span></td>
      <td style="font-weight:600;max-width:160px">${i.name}</td>
      <td style="color:var(--text-2)">${i.category}</td>
      <td><span class="abc-inline ${i.abc_class}">${abcLabel(i.abc_class)}</span></td>
      <td>
        <div class="stock-bar-wrap">
          <div class="stock-bar-bg"><div class="stock-bar-fill ${i.status}" style="width:${pct}%"></div></div>
          <span style="font-weight:700;white-space:nowrap">${fmtNum(i.stock)}</span>
        </div>
      </td>
      <td style="font-weight:700;color:${fcColor}">${fmtNum(i.forecast_total_units)}</td>
      <td style="color:var(--text-2)">${fmtNum(i.reorder_pt)}</td>
      <td style="font-weight:700;color:${daysColor}">${i.days_of_stock <= 999 ? i.days_of_stock+'d' : '∞'}</td>
      <td style="color:var(--text-2)">${i.daily_demand}/day</td>
      <td><span class="status-badge ${i.status}">${statusLabel(i.status)}</span></td>
      <td><button class="btn-restock" onclick="restock('${i.sku}','${i.name}')">Reorder</button></td>
    </tr>`;
  }).join('');
}

function filterTable() {
  const q    = (document.getElementById('skuSearch')?.value || '').toLowerCase();
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
    body.innerHTML = '<div class="alerts-empty">All stocked up — nothing needs attention right now.</div>';
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
      <p style="font-size:0.82rem;color:var(--text-2);margin:0.5rem 0 0.75rem;line-height:1.55">${a.alert_message || ''}</p>
      <div class="alert-meta">
        <span>In stock: <strong>${fmtNum(a.stock)}</strong></span>
        <span>Needed (4 wks): <strong>${fmtNum(a.forecast_total_units)}</strong></span>
        <span>Short by: <strong>${fmtNum(a.shortfall_units)}</strong></span>
        <span class="order-cost">Order cost: ${fmtCur(a.order_cost)}</span>
        <span>Wait time: <strong>${a.lead_time} days</strong></span>
      </div>
      <div class="alert-actions">
        <button class="btn-order-now ${a.status}" onclick="restock('${a.sku}','${a.name}')">
          Order ${fmtNum(a.order_qty)} units
        </button>
      </div>
    </div>`).join('');
}

// ---- Forecast Chart -------------------------------------------------

function populateForecastDropdown(items) {
  const sel = document.getElementById('forecastSku');
  const cur = sel.value;
  sel.innerHTML = '<option value="">Choose a product…</option>' +
    items.map(i => `<option value="${i.sku}" ${i.sku===cur?'selected':''}>${i.sku} — ${i.name}</option>`).join('');
  if (!cur && items.length) loadForecast(items[0].sku);
}

async function loadForecast(sku) {
  if (!sku) return;
  const sel = document.getElementById('forecastSku');
  sel.value = sku;
  try {
    const d = await SalevoraAPI.inventory.forecast(sku);
    const fc = d.forecasts[0];
    if (!fc) return;
    const days    = fc.forecast.map(f=>f.day);
    const stocks  = fc.forecast.map(f=>f.projected_stock);
    const demands = fc.forecast.map(f=>f.demand);
    const plotLayout = {
      paper_bgcolor:'transparent', plot_bgcolor:'transparent',
      font:{ color:'#5C5650', family:'DM Sans,sans-serif', size:11 },
      xaxis:{ gridcolor:'rgba(31,28,24,0.06)', tickfont:{size:10} },
      yaxis:{ gridcolor:'rgba(31,28,24,0.06)', tickprefix:'', tickformat:',d', title:{text:'Units',font:{size:10,color:'#8A837A'}} },
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
        line:{color:'#2F5233',width:2.2,shape:'spline'}, fill:'tozeroy', fillcolor:'rgba(47,82,51,0.08)',
        marker:{size:6,color:'#2F5233',line:{width:1.5,color:'#fff'}},
        hovertemplate:'<b>%{x}</b><br>Stock: %{y:,d} units<extra></extra>',
      },
      {
        x:days, y:demands, type:'bar', name:'Est. Daily Demand',
        marker:{color:'rgba(3,105,161,0.35)'}, yaxis:'y2',
        hovertemplate:'<b>%{x}</b><br>Demand: %{y:.1f}/day<extra></extra>',
      },
    ], {
      ...plotLayout,
      yaxis2:{overlaying:'y', side:'right', showgrid:false, tickformat:'.1f',
               title:{text:'Demand/day',font:{size:10,color:'#556080'}}, range:[0, Math.max(...demands)*3]},
    }, { displayModeBar:false, responsive:true });

    const stockoutDay = fc.stockout_day;
    document.getElementById('forecastMeta').innerHTML = `
      <div class="forecast-meta-item"><div class="forecast-meta-val">${fmtNum(fc.current_stock)}</div><div class="forecast-meta-lab">In stock now</div></div>
      <div class="forecast-meta-item"><div class="forecast-meta-val">${fc.daily_demand}/day</div><div class="forecast-meta-lab">Sold per day</div></div>
      <div class="forecast-meta-item"><div class="forecast-meta-val">${fmtNum(fc.forecast_total_units)}</div><div class="forecast-meta-lab">Expected sales (4 wks)</div></div>
      <div class="forecast-meta-item"><div class="forecast-meta-val" style="color:${stockoutDay?'var(--red)':'var(--green)'}">${stockoutDay || 'More than 7 days'}</div><div class="forecast-meta-lab">May run out around</div></div>`;
  } catch(e) {
    document.getElementById('forecastChart').innerHTML = '<div style="padding:2rem;color:var(--text-3)">Could not load this chart. Please try again.</div>';
  }
}

// ---- ABC Chart ------------------------------------------------------

function renderABC(data) {
  const classes = ['A','B','C'];
  const colors  = ['rgba(47,82,51,0.85)','rgba(3,105,161,0.7)','rgba(90,122,94,0.65)'];
  const counts  = classes.map(c => data.items.filter(i=>i.abc_class===c).length);
  const values  = classes.map(c => data.items.filter(i=>i.abc_class===c).reduce((s,i)=>s+i.annual_value,0));
  Plotly.newPlot('abcChart', [
    {
      labels: classes.map((c,i) => `${c} — ${counts[i]} products`),
      values: values,
      type: 'pie',
      hole: 0.52,
      marker: { colors },
      textinfo: 'label+percent',
      hovertemplate: '<b>Class %{label}</b><br>Annual Value: $%{value:,.0f}<extra></extra>',
      textfont: { size: 12, color: '#1F1C18', family:'DM Sans,sans-serif' },
    }
  ], {
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'transparent',
    font: { color:'#5C5650', family:'DM Sans,sans-serif', size:11 },
    showlegend: false,
    margin: { l:10, r:10, t:15, b:10 },
    annotations: [{
      text: `$${(data.total_annual_value/1000).toFixed(0)}K<br><span style="font-size:9px">Annual</span>`,
      x:0.5, y:0.5, showarrow:false,
      font: { size:16, color:'#1F1C18', family:'Instrument Serif,serif' },
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
      <div class="eoq-unit">items per order</div>
      <div class="eoq-cost">About ${fmtCur(i.eoq * i.cost)} per order</div>
    </div>`).join('');
}

// ---- Restock Action -------------------------------------------------

async function restock(sku, name) {
  try {
    const d = await SalevoraAPI.inventory.restock(sku);
    toast(`Ordered ${d.qty_added} units of "${name}"`, 'success');
    // Highlight button as restocked
    const btns = document.querySelectorAll(`.btn-restock`);
    btns.forEach(b => { if (b.closest('tr')?.querySelector('td span')?.textContent === sku) b.classList.add('restocked'); });
    await fetchAll();
  } catch(e) {
    toast('Could not connect. Make sure the app is running, then try again.', 'error', 5000);
  }
}

// ---- WebSocket ------------------------------------------------------

function connectWS() {
  try {
    ws = new WebSocket(WS);
    ws.onopen = () => {
      setLiveBadge(true);
      toast('Connected — stock updates automatically.', 'success');
    };
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'snapshot') {
        allItems = data.items;
        renderTable(allItems);
        populateCategoryFilter(allItems);
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

function handleInvLogout() {
  clearSession();
  window.location.href = 'index.html';
}

function syncSkuSearch(value) {
  const hidden = document.getElementById('skuSearch');
  if (hidden) hidden.value = value;
  filterTable();
}

window.addEventListener('DOMContentLoaded', async () => {
  const user = await requireAuth(true);
  if (!user) return;
  if (typeof decorateIcons === 'function') decorateIcons();
  if (typeof decorateSidebar === 'function') decorateSidebar();
  if (typeof setSidebarUser === 'function') setSidebarUser(user);
  await fetchAll();
  startCountdown();
  connectWS();
});
