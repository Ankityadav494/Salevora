/* =============================================
   SALEVORA — app.js
   Auth + File Processing + Predictions + Alerts
   ============================================= */

/* ========================
   AUTH — User Store
   ======================== */
const USERS_KEY      = 'salevora_users';
const SESSION_KEY    = 'salevora_session';
let currentFileName  = '';

// SHA-256 password hashing (Web Crypto API — built into every modern browser)
async function hashPassword(pw) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(pw + 'sv_salt_2024'));
  return Array.from(new Uint8Array(buf)).map(b=>b.toString(16).padStart(2,'0')).join('');
}

// IndexedDB store for large datasets (replaces 5 MB localStorage limit)
class SvStore {
  static _db = null;
  static async db() {
    if (this._db) return this._db;
    return this._db = await new Promise((res,rej)=>{
      const r = indexedDB.open('SalevoraBrain',1);
      r.onupgradeneeded = e => e.target.result.createObjectStore('kv');
      r.onsuccess = e => res(e.target.result);
      r.onerror   = e => rej(e.target.error);
    });
  }
  static async get(k) {
    const db = await this.db();
    return new Promise((res,rej)=>{ const r=db.transaction('kv').objectStore('kv').get(k); r.onsuccess=e=>res(e.target.result); r.onerror=e=>rej(e); });
  }
  static async set(k,v) {
    const db = await this.db();
    return new Promise((res,rej)=>{ const t=db.transaction('kv','readwrite'); t.objectStore('kv').put(v,k); t.oncomplete=res; t.onerror=rej; });
  }
  static async del(k) {
    const db = await this.db();
    return new Promise((res,rej)=>{ const t=db.transaction('kv','readwrite'); t.objectStore('kv').delete(k); t.oncomplete=res; t.onerror=rej; });
  }
}

function getUsers()     { return JSON.parse(localStorage.getItem(USERS_KEY) || '[]'); }
function saveUsers(u)   { localStorage.setItem(USERS_KEY, JSON.stringify(u)); }
function getSession()   { return JSON.parse(localStorage.getItem(SESSION_KEY) || 'null'); }
function saveSession(s) { localStorage.setItem(SESSION_KEY, JSON.stringify(s)); }
function clearSession() { localStorage.removeItem(SESSION_KEY); }

// Seed demo account with SHA-256 hash
(async function seedDemo() {
  const users = getUsers();
  if (!users.find(u => u.email === 'demo@salevora.com')) {
    const hash = await hashPassword('demo1234');
    users.push({ name: 'Demo User', email: 'demo@salevora.com', password: hash });
    saveUsers(users);
  }
})();

/* ========================
   AUTH — UI Helpers
   ======================== */
function showTab(tab) {
  document.getElementById('loginForm').style.display    = tab === 'login'    ? '' : 'none';
  document.getElementById('registerForm').style.display = tab === 'register' ? '' : 'none';
  document.getElementById('tabLogin').classList.toggle('active',    tab === 'login');
  document.getElementById('tabRegister').classList.toggle('active', tab === 'register');
  clearAuthMessages();
}

function clearAuthMessages() {
  ['loginError','regError','regSuccess'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.style.display = 'none'; el.textContent = ''; }
  });
}

function showError(id, msg)   { const el = document.getElementById(id); el.textContent = msg; el.style.display = ''; }
function showSuccess(id, msg) { const el = document.getElementById(id); el.textContent = msg; el.style.display = ''; }

function togglePwd(inputId, btn) {
  const input = document.getElementById(inputId);
  if (input.type === 'password') { input.type = 'text'; btn.textContent = '🙈'; }
  else                           { input.type = 'password'; btn.textContent = '👁'; }
}

/* ========================
   AUTH — Login
   ======================== */
async function handleLogin(e) {
  e.preventDefault();
  clearAuthMessages();
  const email    = document.getElementById('loginEmail').value.trim().toLowerCase();
  const password = document.getElementById('loginPassword').value;
  const btn      = document.getElementById('loginSubmit');
  btn.textContent = 'Signing in…'; btn.disabled = true;
  const hash  = await hashPassword(password);
  const users = getUsers();
  let user    = users.find(u => u.email === email && u.password === hash);
  // Backward-compat: migrate old btoa() passwords to SHA-256
  if (!user) {
    const legacy = users.find(u => u.email === email && u.password === btoa(password));
    if (legacy) { legacy.password = hash; saveUsers(users); user = legacy; }
  }
  if (user) {
    saveSession({ name: user.name, email: user.email });
    enterApp(user);
  } else {
    showError('loginError', '❌ Invalid email or password. Try demo@salevora.com / demo1234');
  }
  btn.textContent = 'Sign In'; btn.disabled = false;
}

/* ========================
   AUTH — Register
   ======================== */
async function handleRegister(e) {
  e.preventDefault();
  clearAuthMessages();
  const name     = document.getElementById('regName').value.trim();
  const email    = document.getElementById('regEmail').value.trim().toLowerCase();
  const password = document.getElementById('regPassword').value;
  const confirm  = document.getElementById('regConfirm').value;
  const btn      = document.getElementById('regSubmit');
  if (password !== confirm) { showError('regError', '❌ Passwords do not match.'); return; }
  if (password.length < 6)  { showError('regError', '❌ Password must be at least 6 characters.'); return; }
  const users = getUsers();
  if (users.find(u => u.email === email)) { showError('regError', '❌ An account with this email already exists.'); return; }
  btn.textContent = 'Creating account…'; btn.disabled = true;
  const hash = await hashPassword(password);
  users.push({ name, email, password: hash });
  saveUsers(users);
  saveSession({ name, email });
  showSuccess('regSuccess', '✅ Account created! Signing you in…');
  setTimeout(() => enterApp({ name, email }), 800);
  btn.textContent = 'Create Account'; btn.disabled = false;
}

/* ========================
   APP — Enter / Exit
   ======================== */
function enterApp(user) {
  document.getElementById('authScreen').style.display = 'none';
  document.getElementById('appScreen').style.display  = '';
  document.getElementById('navUser').textContent = `👤 ${user.name}`;

  // Re-inject alerts section if results are showing
  maybeAddAlertSection();
  checkRestoreData();
  toast(`Welcome back, ${user.name.split(' ')[0]}! 👋`, 'success');
}

function handleLogout() {
  clearSession();
  document.getElementById('authScreen').style.display = '';
  document.getElementById('appScreen').style.display  = 'none';
  // Clear file state
  rawData = []; mappedData = []; weeklyData = [];
  document.getElementById('resultsWrap').style.display = 'none';
  document.getElementById('colConfig').style.display = 'none';
  document.getElementById('uploadStatus').className = 'upload-status';
  document.getElementById('uploadStatus').textContent = '';
  toast('Signed out successfully.', 'success');
}

function scrollTop() { window.scrollTo({ top: 0, behavior: 'smooth' }); }

// Auto-login if session active
window.addEventListener('DOMContentLoaded', () => {
  const session = getSession();
  if (session) { enterApp(session); }
});

/* ========================
   DATA PERSISTENCE (IndexedDB — per-user)
   ======================== */
async function saveToStorage() {
  if (!weeklyData.length) return;
  const key = 'sv_data_' + (getSession()?.email || 'anon');
  try {
    await SvStore.set(key, {
      fileName: currentFileName || 'dataset',
      colMap,
      savedAt: new Date().toISOString(),
      weeklyData: weeklyData.map(w => ({ d: w.date.getTime(), s: w.sales, r: w.revenue })),
      mappedData: rawMappedData.slice(0, 5000).map(r => ({ d: r.date.getTime(), s: r.sales, rv: r.revenue, c: r.category }))
    });
  } catch(e) { console.warn('IndexedDB save failed:', e); }
}

async function checkRestoreData() {
  try {
    const key   = 'sv_data_' + (getSession()?.email || 'anon');
    const saved = await SvStore.get(key);
    if (!saved?.weeklyData?.length) return;
    const banner = document.getElementById('restoreBanner');
    if (!banner) return;
    const d = new Date(saved.savedAt);
    banner.style.display = '';
    document.getElementById('restoreInfo').textContent =
      `“${saved.fileName}” · ${saved.mappedData?.length?.toLocaleString() || '?'} rows · saved ${d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'})} ${d.toLocaleDateString()}`;
  } catch(e) {}
}

async function restoreFromStorage() {
  try {
    const key   = 'sv_data_' + (getSession()?.email || 'anon');
    const saved = await SvStore.get(key);
    if (!saved?.weeklyData?.length) { toast('No saved data found.', 'error'); return; }
    rawMappedData = saved.mappedData.map(r => ({ date: new Date(r.d), sales: r.s, revenue: r.rv, category: r.c }));
    mappedData    = [...rawMappedData];
    rawWeeklyData = saved.weeklyData.map(w => ({ date: new Date(w.d), sales: w.s, revenue: w.r }));
    weeklyData    = [...rawWeeklyData];
    colMap        = saved.colMap;
    currentFileName = saved.fileName;
    document.getElementById('restoreBanner').style.display = 'none';
    document.getElementById('resultsWrap').style.display = '';
    document.getElementById('colConfig').style.display = 'none';
    buildKPIs(); buildTrendChart(); buildCategoryChart(); buildMonthlyChart();
    buildTopPerformers(); buildYoYChart(); buildProductChart(); updateForecast(); buildTable(); maybeAddAlertSection();
    document.getElementById('kpiSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    toast(`✅ Restored “${saved.fileName}” for ${getSession()?.email || 'you'}!`, 'success');
  } catch(e) { toast('Could not restore session data.', 'error'); }
}

async function clearStoredData() {
  const key = 'sv_data_' + (getSession()?.email || 'anon');
  await SvStore.del(key);
  const b = document.getElementById('restoreBanner');
  if (b) b.style.display = 'none';
  toast('Session data cleared.', 'info');
}

/* ========================
   GOAL TRACKER
   ======================== */
function buildGoalTracker() {
  const el = document.getElementById('goalTracker');
  if (!el || !mappedData.length) { if (el) el.style.display = 'none'; return; }
  const goal = parseFloat(localStorage.getItem('sv_goal') || 0);
  const endMs = mappedData[mappedData.length - 1].date.getTime();
  const cur   = mappedData.filter(r => (endMs - r.date.getTime()) / 86400000 <= 30).reduce((s, r) => s + r.sales, 0);
  const pct   = goal > 0 ? Math.min(100, cur / goal * 100) : 0;
  const cls   = pct >= 80 ? 'on-track' : pct >= 50 ? 'at-risk' : goal > 0 ? 'behind' : 'no-goal';
  const statusTxt = !goal ? '❓ Set a target to track progress' : pct >= 80 ? '🟢 On Track' : pct >= 50 ? '🟡 At Risk' : '🔴 Behind Target';
  const now = new Date();
  const daysLeft = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate() - now.getDate();
  el.style.display = '';
  el.innerHTML = `
    <div class="goal-card">
      <div class="goal-header">
        <div>
          <div class="goal-title">🎯 Monthly Sales Goal</div>
          <div class="goal-badge ${cls}">${statusTxt}</div>
        </div>
        <div class="goal-input-group">
          <label class="goal-input-label">Set Target ($)</label>
          <input type="number" class="goal-input" value="${goal || ''}" placeholder="e.g. 500000"
            onchange="localStorage.setItem('sv_goal',this.value||0);buildGoalTracker();" />
        </div>
      </div>
      <div class="goal-bar-wrap"><div class="goal-bar ${cls}" style="width:${pct.toFixed(1)}%"></div></div>
      <div class="goal-meta">
        <div class="goal-stat"><div class="goal-stat-val">${fmtCur(cur)}</div><div class="goal-stat-lab">Last 30 Days</div></div>
        <div class="goal-stat"><div class="goal-stat-val">${goal ? fmtCur(goal) : '—'}</div><div class="goal-stat-lab">Target</div></div>
        <div class="goal-stat"><div class="goal-stat-val">${goal ? pct.toFixed(1) + '%' : '—'}</div><div class="goal-stat-lab">Achieved</div></div>
        <div class="goal-stat"><div class="goal-stat-val">${daysLeft}</div><div class="goal-stat-lab">Days Left</div></div>
      </div>
    </div>`;
}

/* ========================
   PDF EXPORT
   ======================== */
async function exportPDF() {
  if (!weeklyData.length) { toast('Load data first before exporting.', 'warning'); return; }
  toast('Preparing report\u2026 \ud83d\udcc4', 'info', 5000);

  // Capture chart images from Plotly
  let trendImg = '', forecastImg = '';
  try { trendImg = await Plotly.toImage('trendChart', {format:'png',width:1000,height:380,scale:1.5}); } catch(e){}
  try { forecastImg = await Plotly.toImage('forecastChart', {format:'png',width:1000,height:380,scale:1.5}); } catch(e){}

  // Read KPI cards from DOM
  const kpiCards = [...document.querySelectorAll('.kpi-card')].map(c=>`
    <div class="kpi">
      <div class="kpi-l">${c.querySelector('.kpi-label')?.innerText||''}</div>
      <div class="kpi-v">${c.querySelector('.kpi-value')?.innerText||''}</div>
      <div class="kpi-t">${(c.querySelector('.kpi-trend-up')||c.querySelector('.kpi-trend-down')||c.querySelector('.kpi-sub'))?.innerText||''}</div>
    </div>`).join('');

  // Top performers rows
  const perfRows = [...document.querySelectorAll('.perf-card')].map(c=>`
    <tr>
      <td>${c.querySelector('.perf-rank')?.innerText||''}</td>
      <td>${c.querySelector('.perf-name')?.innerText||''}</td>
      <td>${[...c.querySelectorAll('.perf-stats span')].map(s=>s.innerText).join(' &middot; ')}</td>
    </tr>`).join('');

  const insight = document.getElementById('insightBanner')?.innerText?.trim() || '';
  const summary = document.getElementById('summaryText')?.innerText?.trim() || '';
  const today   = new Date().toLocaleDateString('en-US',{dateStyle:'full'});
  const filename = `Salevora_Report_${new Date().toISOString().slice(0,10)}`;

  const html = `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>${filename}</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:Arial,Helvetica,sans-serif;color:#111827;background:#fff;font-size:13px}
  .header{background:linear-gradient(135deg,#0a0d24 0%,#1e1b4b 100%);color:#fff;padding:28px 40px;display:flex;justify-content:space-between;align-items:center}
  .logo{font-size:26px;font-weight:900;letter-spacing:-0.5px}.logo em{color:#a78bfa;font-style:normal}
  .header-sub{font-size:11px;opacity:.7;margin-top:5px}
  .header-right{text-align:right;font-size:11px;opacity:.75;line-height:1.7}
  .body{padding:30px 40px}
  .insight{background:#eef2ff;border-left:4px solid #6366f1;padding:11px 15px;margin-bottom:20px;border-radius:4px;font-size:12px;color:#3730a3;font-weight:500}
  .summary{font-size:11px;color:#6b7280;margin-bottom:22px}
  h2{font-size:14px;font-weight:700;color:#111827;border-bottom:2px solid #6366f1;padding-bottom:5px;margin-bottom:12px;margin-top:24px}
  .kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:8px}
  .kpi{background:#f9fafb;border:1px solid #e5e7eb;border-top:3px solid #6366f1;border-radius:6px;padding:11px}
  .kpi-l{font-size:9px;text-transform:uppercase;letter-spacing:.07em;color:#9ca3af;margin-bottom:4px}
  .kpi-v{font-size:17px;font-weight:800;color:#111827}
  .kpi-t{font-size:10px;color:#6b7280;margin-top:3px}
  .chart{width:100%;height:auto;border-radius:6px;border:1px solid #e5e7eb;margin-bottom:6px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  thead th{background:#6366f1;color:#fff;padding:8px 10px;text-align:left;font-size:11px}
  tbody td{padding:8px 10px;border-bottom:1px solid #f3f4f6}
  tbody tr:nth-child(even) td{background:#f9fafb}
  .footer{margin-top:30px;padding-top:12px;border-top:1px solid #e5e7eb;display:flex;justify-content:space-between;font-size:10px;color:#9ca3af}
  @media print{
    body{-webkit-print-color-adjust:exact;print-color-adjust:exact}
    h2{page-break-after:avoid}
    .no-break{page-break-inside:avoid}
  }
</style></head><body>
<div class="header">
  <div><div class="logo">&#9889; Sale<em>vora</em></div><div class="header-sub">Sales Intelligence Report &nbsp;&middot;&nbsp; Confidential</div></div>
  <div class="header-right"><strong>${today}</strong><br>Generated by Salevora</div>
</div>
<div class="body">
  ${insight?`<div class="insight">${insight}</div>`:''}
  ${summary?`<p class="summary">&#128202; ${summary}</p>`:''}
  <h2>Key Performance Indicators</h2>
  <div class="kpi-grid no-break">${kpiCards}</div>
  ${trendImg?`<h2>Historical Sales Trend + Anomaly Detection</h2><img class="chart no-break" src="${trendImg}" />`:''}
  ${forecastImg?`<h2>Sales Forecast &mdash; Weighted Ensemble Model</h2><img class="chart no-break" src="${forecastImg}" />`:''}
  ${perfRows?`<h2>Top Performers by Category</h2><table class="no-break"><thead><tr><th>Rank</th><th>Category</th><th>Stats</th></tr></thead><tbody>${perfRows}</tbody></table>`:''}
  <div class="footer">
    <span>&#9889; Salevora Sales Intelligence</span>
    <span>All data processed locally &mdash; never leaves your browser</span>
    <span>${new Date().toLocaleDateString()}</span>
  </div>
</div>
<script>window.addEventListener('load',function(){setTimeout(function(){window.print();},500);})<\/script>
</body></html>`;

  const win = window.open('', '_blank', 'width=900,height=700');
  if (!win) { toast('Pop-up blocked! Allow pop-ups for this site and try again.', 'error', 6000); return; }
  win.document.write(html);
  win.document.close();
  toast('\ud83d\udcc4 Report ready! Choose "Save as PDF" in the print dialog.', 'success', 6000);
}

/* ========================
   TOAST
   ======================== */
function toast(msg, type = 'success', duration = 3500) {
  let container = document.getElementById('toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    document.body.appendChild(container);
  }
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  container.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transform = 'translateX(20px)'; t.style.transition = 'all 0.3s'; setTimeout(() => t.remove(), 350); }, duration);
}

/* ========================
   NAVBAR NOTIFICATIONS
   ======================== */
let notifCount = 0;

function addNavNotification(title, sub, type = 'info') {
  const list  = document.getElementById('notifList');
  const badge = document.getElementById('notifBadge');
  if (!list || !badge) return;

  // Remove empty state
  const empty = list.querySelector('.notif-empty');
  if (empty) empty.remove();

  // Create item
  const item = document.createElement('div');
  item.className = 'notif-item';
  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  item.innerHTML = `
    <span class="notif-dot ${type}"></span>
    <div class="notif-item-body">
      <div class="notif-item-title">${title}</div>
      <div class="notif-item-sub">${sub} &middot; ${timeStr}</div>
    </div>`;

  // Prepend so newest shows first
  list.insertBefore(item, list.firstChild);

  // Update badge
  notifCount++;
  badge.textContent = notifCount > 99 ? '99+' : notifCount;
  badge.style.display = 'flex';

  // Shake the bell
  const bell = document.getElementById('notifBellBtn');
  if (bell) { bell.style.animation = 'none'; void bell.offsetWidth; bell.style.animation = 'bellShake 0.5s ease'; }
}

function toggleNotifDropdown() {
  const dd = document.getElementById('notifDropdown');
  if (!dd) return;
  const isOpen = dd.style.display !== 'none';
  dd.style.display = isOpen ? 'none' : '';
  // Reset badge when opened
  if (!isOpen) {
    notifCount = 0;
    const badge = document.getElementById('notifBadge');
    if (badge) badge.style.display = 'none';
  }
}

function clearAllNotifications() {
  const list = document.getElementById('notifList');
  if (list) list.innerHTML = '<div class="notif-empty">No notifications yet</div>';
  notifCount = 0;
  const badge = document.getElementById('notifBadge');
  if (badge) badge.style.display = 'none';
  document.getElementById('notifDropdown').style.display = 'none';
}

// Close dropdown when clicking outside
document.addEventListener('click', e => {
  const wrap = document.getElementById('notifBellWrap');
  const dd   = document.getElementById('notifDropdown');
  if (wrap && dd && !wrap.contains(e.target)) dd.style.display = 'none';
});

/* ========================
   FILE UPLOAD
   ======================== */
let rawData = [], mappedData = [], weeklyData = [], forecastValues = [], allColumns = [];
let rawMappedData = [], rawWeeklyData = []; // Originals preserved for date-range filter
let colMap = { date:'', sales:'', revenue:'', category:'' };
let filteredRows = [], currentPage = 1;
const PAGE_SIZE = 25;

/* ========================
   DATE RANGE FILTER
   ======================== */
function applyDateFilter() {
  const sv = document.getElementById('filterStart')?.value;
  const ev = document.getElementById('filterEnd')?.value;
  const s  = sv ? new Date(sv) : null;
  const e  = ev ? new Date(ev) : null;
  mappedData  = rawMappedData.filter(r => (!s || r.date >= s) && (!e || r.date <= e));
  weeklyData  = aggregateWeekly(mappedData);
  if (!mappedData.length) { toast('No data in selected range. Try a wider range.', 'warning'); return; }
  redrawAll();
  toast(`Filtered to ${mappedData.length.toLocaleString()} records in range.`, 'info');
}

function resetDateFilter() {
  document.getElementById('filterStart').value = '';
  document.getElementById('filterEnd').value   = '';
  mappedData = [...rawMappedData];
  weeklyData = [...rawWeeklyData];
  redrawAll();
  toast('Date filter removed — showing all data.', 'info');
}

function redrawAll() {
  buildKPIs(); buildTrendChart(); buildCategoryChart(); buildMonthlyChart();
  buildTopPerformers(); buildYoYChart(); buildProductChart(); updateForecast(); buildTable();
}

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag-over'); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  setUploadStatus('info', `Reading "${file.name}"…`);

  if (ext === 'csv') {
    Papa.parse(file, {
      header: true, skipEmptyLines: true,
      complete: r => processRaw(r.data, file.name),
      error:    err => setUploadStatus('error', 'CSV parse error: ' + err.message),
    });
  } else if (['xlsx','xls'].includes(ext)) {
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const wb = XLSX.read(ev.target.result, { type: 'binary', cellDates: true });
        const ws = wb.Sheets[wb.SheetNames[0]];
        processRaw(XLSX.utils.sheet_to_json(ws, { defval: '' }), file.name);
      } catch (err) { setUploadStatus('error', 'Excel parse error: ' + err.message); }
    };
    reader.readAsBinaryString(file);
  } else {
    setUploadStatus('error', `Unsupported file type ".${ext}". Please use a CSV or Excel file.`);
  }
}

function setUploadStatus(type, msg) {
  const el = document.getElementById('uploadStatus');
  el.className = 'upload-status ' + type;
  el.textContent = msg;
}

/* ========================
   SAMPLE DATA LOADER
   ======================== */
function loadSampleData() {
  const cats = ['Electronics','Clothing','Food & Bev','Sports','Home & Garden'];
  const rows = [];
  const base = { Electronics:45000, Clothing:32000, 'Food & Bev':28000, Sports:18000, 'Home & Garden':14000 };
  const start = new Date('2022-01-03');
  for (let w = 0; w < 104; w++) {
    const d = new Date(start); d.setDate(d.getDate() + w * 7);
    const seasonal = 1 + 0.45 * Math.sin((d.getMonth() - 2) * Math.PI / 6);
    const trend    = 1 + w * 0.004;
    cats.forEach(cat => {
      const s = Math.max(500, Math.round(base[cat] * seasonal * trend * (0.75 + Math.random() * 0.5)));
      rows.push({ date: d.toISOString().slice(0,10), sales: s, revenue: Math.round(s * (1.15 + Math.random()*0.2)), category: cat });
    });
  }
  rawData = rows; allColumns = ['date','sales','revenue','category'];
  colMap = { date:'date', sales:'sales', revenue:'revenue', category:'category' };
  document.getElementById('colConfig').style.display = 'none';
  setUploadStatus('success', `✅ Sample dataset loaded — ${rows.length.toLocaleString()} rows · 2 years · ${cats.length} categories · 104 weeks`);
  setTimeout(() => runPrediction(), 200);
}

function processRaw(data, filename) {
  if (!data || !data.length) { setUploadStatus('error', 'File is empty.'); return; }
  rawData = data;
  currentFileName = filename;
  allColumns = Object.keys(data[0]);

  const lc = allColumns.map(c => c.toLowerCase());
  colMap.date     = allColumns.find((_,i) => lc[i].includes('date')) || '';
  colMap.sales    = allColumns.find((_,i) => lc[i].includes('sale') || lc[i].includes('amount') || lc[i].includes('revenue') && !lc[i].includes('date')) || '';
  colMap.revenue  = allColumns.find((_,i) => lc[i].includes('revenue') || lc[i].includes('total')) || '';
  colMap.category = allColumns.find((_,i) => lc[i].includes('category') || lc[i].includes('cat') || lc[i].includes('type') || lc[i].includes('segment')) || '';
  if (!colMap.sales && colMap.revenue) { colMap.sales = colMap.revenue; }

  setUploadStatus('success', `✅ Loaded "${filename}" — ${data.length.toLocaleString()} rows, ${allColumns.length} columns detected.`);
  buildColConfig();
}

function buildColConfig() {
  const wrap = document.getElementById('colConfig');
  const grid = document.getElementById('colConfigGrid');
  wrap.style.display = '';

  const fields = [
    { key:'date',     label:'Date Column *'   },
    { key:'sales',    label:'Sales Column *'  },
    { key:'revenue',  label:'Revenue Column'  },
    { key:'category', label:'Category Column' },
  ];

  grid.innerHTML = fields.map(f => `
    <div class="col-config-item">
      <label>${f.label}</label>
      <select id="col_${f.key}" onchange="colMap['${f.key}']=this.value">
        ${f.key === 'revenue' || f.key === 'category' ? '<option value="">— None —</option>' : ''}
        ${allColumns.map(c => `<option value="${c}" ${colMap[f.key]===c?'selected':''}>${c}</option>`).join('')}
      </select>
    </div>`).join('');
}

/* ========================
   RUN PREDICTION
   ======================== */
function runPrediction() {
  // Only read from DOM selectors if the column config UI is showing
  const colConfigVisible = document.getElementById('colConfig').style.display !== 'none';
  if (colConfigVisible) {
    colMap.date     = document.getElementById('col_date')?.value    || colMap.date;
    colMap.sales    = document.getElementById('col_sales')?.value   || colMap.sales;
    colMap.revenue  = document.getElementById('col_revenue')?.value  || colMap.revenue;
    colMap.category = document.getElementById('col_category')?.value || colMap.category;
  }

  if (!colMap.date || !colMap.sales) {
    setUploadStatus('error', 'Please select both a Date and a Sales column.'); return;
  }

  mappedData  = rawData.map(row => {
    const d = new Date(row[colMap.date]);
    if (isNaN(d.getTime())) return null;
    const sales    = parseFloat(String(row[colMap.sales]  || 0).replace(/[$,]/g,'')) || 0;
    const revenue  = colMap.revenue ? (parseFloat(String(row[colMap.revenue] || 0).replace(/[$,]/g,'')) || sales) : sales;
    const category = colMap.category ? String(row[colMap.category] || 'Uncategorised') : 'Uncategorised';
    // Detect product/SKU column automatically
    const productCol = allColumns.find(c => /product|sku|item|name/i.test(c));
    const product  = productCol ? String(row[productCol] || 'Unknown') : null;
    return { date: d, sales, revenue, category, product };
  }).filter(Boolean).sort((a,b) => a.date - b.date);

  if (!mappedData.length) {
    setUploadStatus('error', 'No valid date rows found. Check your date column (use YYYY-MM-DD).'); return;
  }

  // Preserve originals for date-range filter and YoY chart
  rawMappedData = [...mappedData];
  weeklyData    = aggregateWeekly(mappedData);
  rawWeeklyData = [...weeklyData];

  // Init date filter inputs span
  const allDates = mappedData.map(r => r.date);
  const minD = new Date(Math.min(...allDates)).toISOString().slice(0,10);
  const maxD = new Date(Math.max(...allDates)).toISOString().slice(0,10);
  const fs = document.getElementById('filterStart'), fe = document.getElementById('filterEnd');
  if (fs) { fs.min = minD; fs.max = maxD; fs.value = ''; }
  if (fe) { fe.min = minD; fe.max = maxD; fe.value = ''; }
  const df = document.getElementById('dateFilter'); if (df) df.style.display = '';

  document.getElementById('resultsWrap').style.display = '';
  buildKPIs();
  buildTrendChart();
  buildCategoryChart();
  buildMonthlyChart();
  buildTopPerformers();
  buildYoYChart();
  buildProductChart();
  updateForecast();
  buildTable();
  maybeAddAlertSection();

  document.getElementById('kpiSection').scrollIntoView({ behavior:'smooth', block:'start' });
  toast('Analysis complete! Scroll down to see your forecasts.', 'success');
  saveToStorage();
}

/* ========================
   AGGREGATION
   ======================== */
function aggregateWeekly(rows) {
  const map = new Map();
  rows.forEach(r => {
    const d = new Date(r.date);
    const day = d.getDay() || 7;
    d.setDate(d.getDate() - day + 1);
    const key = d.toISOString().slice(0,10);
    if (!map.has(key)) map.set(key, { date: new Date(d), sales:0, revenue:0 });
    const e = map.get(key);
    e.sales   += r.sales;
    e.revenue += r.revenue;
  });
  return [...map.values()].sort((a,b) => a.date - b.date);
}

/* ========================
   KPIs
   ======================== */
function buildKPIs() {
  const total = mappedData.reduce((s,r)=>s+r.revenue,0);
  const sales = mappedData.reduce((s,r)=>s+r.sales,0);
  const rows  = mappedData.length;
  const start = mappedData[0].date;
  const end   = mappedData[mappedData.length-1].date;
  const days  = Math.max(1,(end-start)/86400000+1);
  const endMs = end.getTime();
  const last4 = weeklyData.slice(-4).reduce((s,w)=>s+w.sales,0);
  const prev4 = weeklyData.slice(-8,-4).reduce((s,w)=>s+w.sales,0);
  const delta = prev4?(last4-prev4)/prev4*100:0;
  const cats  = [...new Set(mappedData.map(r=>r.category))];

  // MoM: last 30 days vs prior 30 days
  const last30 = mappedData.filter(r=>(endMs-r.date.getTime())/86400000<=30).reduce((s,r)=>s+r.sales,0);
  const prev30 = mappedData.filter(r=>{const d=(endMs-r.date.getTime())/86400000;return d>30&&d<=60;}).reduce((s,r)=>s+r.sales,0);
  const mom    = prev30>0?(last30-prev30)/prev30*100:null;

  // YoY: last 365 days vs prior 365 days
  const last365 = mappedData.filter(r=>(endMs-r.date.getTime())/86400000<=365).reduce((s,r)=>s+r.sales,0);
  const prev365 = mappedData.filter(r=>{const d=(endMs-r.date.getTime())/86400000;return d>365&&d<=730;}).reduce((s,r)=>s+r.sales,0);
  const yoy     = prev365>0?(last365-prev365)/prev365*100:null;

  // Best category
  const catMap={}; mappedData.forEach(r=>{catMap[r.category]=(catMap[r.category]||0)+r.sales;});
  const bestCat = colMap.category?Object.entries(catMap).sort((a,b)=>b[1]-a[1])[0]:null;

  // AI insight banner
  const parts=[];
  if(mom!=null) parts.push(`Sales ${mom>=0?'grew':'fell'} <strong>${Math.abs(mom).toFixed(1)}%</strong> month-over-month`);
  if(yoy!=null) parts.push(`${yoy>=0?'up':'down'} <strong>${Math.abs(yoy).toFixed(1)}%</strong> year-over-year`);
  if(bestCat)   parts.push(`<strong>${bestCat[0]}</strong> is your top category`);
  const ib=document.getElementById('insightBanner');
  if(ib){ib.style.display=parts.length?'':'none'; ib.innerHTML='🎯 '+parts.join(' · ')+'.';}

  document.getElementById('summaryText').textContent =
    `${rows.toLocaleString()} records · ${fmtDate(start)} → ${fmtDate(end)} · ${weeklyData.length} weeks · ${cats.length} categories`;

  document.getElementById('kpiGrid').innerHTML = [
    { icon:'💰', label:'Total Revenue',  value:fmtCur(total),         sub:`${rows.toLocaleString()} records` },
    { icon:'📈', label:'Total Sales',    value:fmtCur(sales),         sub:`${weeklyData.length} weeks` },
    { icon:'📊', label:'Avg Daily',      value:fmtCur(sales/(days||1)),sub:`Over ${Math.round(days)} days` },
    { icon:'🔄', label:'Last 4 Wks',     value:fmtCur(last4),         trend:delta },
    mom!=null&&{ icon:'📅', label:'MoM Change',   value:fmtCur(last30),        trend:mom },
    yoy!=null&&{ icon:'📆', label:'YoY Growth',   value:fmtCur(last365),       trend:yoy },
    bestCat  &&{ icon:'🏆', label:'Top Category', value:bestCat[0],            sub:fmtCur(bestCat[1])+' in sales' },
    { icon:'📅', label:'Period Start',  value:fmtDate(start),        sub:'Earliest record' },
    { icon:'📆', label:'Period End',    value:fmtDate(end),          sub:'Latest record' },
    { icon:'🏷️', label:'Categories',  value:cats.length,           sub:cats.slice(0,3).join(', ')+(cats.length>3?'…':'') },
  ].filter(Boolean).map(k=>`
    <div class="kpi-card">
      <span class="kpi-icon">${k.icon}</span>
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      ${k.trend!=null
        ?`<div class="${k.trend>=0?'kpi-trend-up':'kpi-trend-down'}">${k.trend>=0?'▲':'▼'} ${Math.abs(k.trend).toFixed(1)}% vs prior</div>`
        :`<div class="kpi-sub">${k.sub||''}</div>`}
    </div>`).join('');

  if(cats.length&&colMap.category){
    document.getElementById('catsPills').innerHTML=cats.map(c=>`<span class="cat-pill">${c}</span>`).join('');
    document.getElementById('catsWrap').style.display='flex';
  }
  buildGoalTracker();
}

/* ========================
   CHARTS
   ======================== */
function buildTrendChart() {
  const dates = weeklyData.map(w=>w.date);
  const sales = weeklyData.map(w=>w.sales);
  const ma4   = sales.map((_,i)=>i>=3?avg(sales.slice(i-3,i+1)):null);
  // Anomaly detection (IQR method)
  const sorted=[...sales].sort((a,b)=>a-b);
  const q1=sorted[Math.floor(sorted.length*0.25)],q3=sorted[Math.floor(sorted.length*0.75)],iqr=q3-q1;
  const anomDates=[],anomVals=[];
  dates.forEach((d,i)=>{ if(sales[i]<q1-1.5*iqr||sales[i]>q3+1.5*iqr){anomDates.push(d);anomVals.push(sales[i]);} });
  if(anomDates.length) addNavNotification('⚠️ Anomaly Detected',`${anomDates.length} unusual week(s) found in your sales data.`,'warning');
  Plotly.newPlot('trendChart',[
    { x:dates,y:sales,type:'scatter',mode:'lines',name:'Weekly Sales',
      line:{color:'#6366f1',width:2.2,shape:'spline'},fill:'tozeroy',fillcolor:'rgba(99,102,241,0.07)',
      hovertemplate:'<b>%{x|%b %d, %Y}</b><br>$%{y:,.0f}<extra></extra>' },
    { x:dates,y:ma4,type:'scatter',mode:'lines',name:'4-Wk Moving Avg',
      line:{color:'#a78bfa',width:1.8,dash:'dot'},hovertemplate:'<b>%{x|%b %d, %Y}</b><br>MA: $%{y:,.0f}<extra></extra>' },
    anomDates.length?{ x:anomDates,y:anomVals,type:'scatter',mode:'markers',name:'⚠️ Anomaly',
      marker:{size:11,color:'#ef4444',symbol:'diamond',line:{width:2,color:'#fff'}},
      hovertemplate:'<b>⚠️ Anomaly %{x|%b %d, %Y}</b><br>$%{y:,.0f}<extra></extra>' }:null,
  ].filter(Boolean),{ ...plotLayout('Weekly Sales + Anomaly Detection'), height:330 },plotCfg());
}

function buildCategoryChart() {
  if (!colMap.category) {
    document.getElementById('catChart').innerHTML = '<div style="color:var(--text-3);padding:2rem;font-size:0.85rem">No category column selected.</div>';
    return;
  }
  const map = {};
  mappedData.forEach(r => { map[r.category] = (map[r.category]||0)+r.revenue; });
  const sorted = Object.entries(map).sort((a,b)=>b[1]-a[1]);
  const colors = ['#6366f1','#8b5cf6','#06b6d4','#10b981','#f97316','#eab308'];
  Plotly.newPlot('catChart',[{
    type:'bar',orientation:'h',
    x:sorted.map(e=>e[1]),y:sorted.map(e=>e[0]),
    marker:{color:sorted.map((_,i)=>colors[i%colors.length])},
    hovertemplate:'<b>%{y}</b><br>$%{x:,.0f}<extra></extra>',
  }],{ ...plotLayout(''), height:250, margin:{l:90,r:10,t:10,b:30} },plotCfg());
}

function buildMonthlyChart() {
  const map = {};
  mappedData.forEach(r => {
    const k = r.date.toLocaleString('default',{year:'numeric',month:'short'});
    map[k] = (map[k]||0)+r.sales;
  });
  const vals = Object.values(map);
  const mean = avg(vals);
  Plotly.newPlot('monthChart',[{
    type:'bar',x:Object.keys(map),y:vals,
    marker:{color:vals.map(v=>v>=mean?'rgba(99,102,241,0.85)':'rgba(139,92,246,0.5)')},
    hovertemplate:'<b>%{x}</b><br>$%{y:,.0f}<extra></extra>',
  }],{ ...plotLayout(''), height:250, bargap:0.25, margin:{l:50,r:10,t:10,b:60} },plotCfg());
}

function buildTopPerformers() {
  const el = document.getElementById('topPerfSection');
  if (el) el.remove();
  if (!colMap.category||!mappedData.length) return;
  const map={};
  mappedData.forEach(r=>{ if(!map[r.category])map[r.category]={sales:0,count:0}; map[r.category].sales+=r.sales; map[r.category].count++; });
  const sorted=Object.entries(map).sort((a,b)=>b[1].sales-a[1].sales);
  const total=sorted.reduce((s,[,v])=>s+v.sales,0);
  const colors=['#6366f1','#8b5cf6','#06b6d4','#10b981','#f97316'];
  const section=document.createElement('section');
  section.id='topPerfSection'; section.className='app-section';
  section.innerHTML=`
    <div class="container">
      <div class="section-label">Performance</div>
      <h2 class="section-title">Top <span class="gradient-text">Performers</span></h2>
      <p class="section-sub">Ranked by total sales contribution. Instantly see which segments are driving your revenue.</p>
      <div class="perf-grid">
        ${sorted.map(([cat,v],i)=>{ const pct=(v.sales/total*100).toFixed(1); return `
          <div class="perf-card" style="--rank-color:${colors[i%colors.length]}">
            <div class="perf-rank">#${i+1}</div>
            <div class="perf-body">
              <div class="perf-name">${cat}</div>
              <div class="perf-bar-wrap"><div class="perf-bar" style="width:${pct}%;background:${colors[i%colors.length]}"></div></div>
              <div class="perf-stats"><span>${fmtCur(v.sales)}</span><span>${v.count.toLocaleString()} records</span><span style="color:${colors[i%colors.length]};font-weight:700">${pct}%</span></div>
            </div>
            <div class="perf-pct" style="color:${colors[i%colors.length]}">${pct}%</div>
          </div>`;}).join('')}
      </div>
    </div>`;
  const fcast=document.getElementById('forecastSection');
  if(fcast) fcast.parentNode.insertBefore(section,fcast);
}

function downloadChart(id) {
  Plotly.downloadImage(id,{format:'png',width:1400,height:600,filename:'salevora_'+id,scale:2});
  toast('Chart downloading as PNG 📸','success');
}

/* ========================
   FORECAST ENGINE
   ======================== */
function updateForecast() {
  if (weeklyData.length < 4) return;
  const weeks   = parseInt(document.getElementById('horizonSelect').value);
  const smooth  = document.getElementById('smoothSelect').value;
  const alpha   = smooth==='light'?0.4:smooth==='heavy'?0.1:0.22;
  const sales   = weeklyData.map(w=>w.sales);
  const n       = sales.length;
  const xs      = sales.map((_,i)=>i);
  const xM=avg(xs),yM=avg(sales);
  const slope   = xs.reduce((s,x,i)=>s+(x-xM)*(sales[i]-yM),0)/xs.reduce((s,x)=>s+(x-xM)**2,0);
  const icept   = yM-slope*xM;
  let ema = sales[0];
  sales.forEach(v=>{ema=alpha*v+(1-alpha)*ema;});
  forecastValues = Array.from({length:weeks},(_,i)=>Math.max(0, 0.6*(icept+slope*(n+i))+0.4*ema));
  const lastDate = weeklyData[weeklyData.length-1].date;
  const fDates   = forecastValues.map((_,i)=>{ const d=new Date(lastDate); d.setDate(d.getDate()+(i+1)*7); return d; });
  const lower    = forecastValues.map(v=>v*0.88);
  const upper    = forecastValues.map(v=>v*1.12);

  Plotly.newPlot('forecastChart',[
    { x:weeklyData.map(w=>w.date),y:sales,type:'scatter',mode:'lines',name:'Historical',
      line:{color:'#6366f1',width:2.2,shape:'spline'},fill:'tozeroy',fillcolor:'rgba(99,102,241,0.06)',
      hovertemplate:'<b>%{x|%b %d, %Y}</b><br>$%{y:,.0f}<extra></extra>' },
    { x:[...fDates,...fDates.slice().reverse()],y:[...upper,...lower.slice().reverse()],
      fill:'toself',fillcolor:'rgba(249,115,22,0.09)',line:{color:'transparent'},hoverinfo:'skip',name:'±12% Band' },
    { x:fDates,y:forecastValues,type:'scatter',mode:'lines+markers',name:'Forecast',
      line:{color:'#f97316',width:2.2,dash:'dot'},marker:{size:6,color:'#f97316',line:{width:1.5,color:'#fff'}},
      hovertemplate:'<b>%{x|%b %d, %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>' },
    { x:[weeklyData[weeklyData.length-1].date,fDates[0]],y:[sales[sales.length-1],forecastValues[0]],
      mode:'lines',line:{color:'#f97316',width:1,dash:'dot'},showlegend:false,hoverinfo:'skip' },
  ],{ ...plotLayout('Sales Forecast — Weighted Ensemble (Linear Trend + EMA)'), height:370 },plotCfg());

  const trendPct = forecastValues[0]?((forecastValues[forecastValues.length-1]-forecastValues[0])/forecastValues[0]*100):0;

  // MAPE: hold-out last 4 weeks to measure forecast accuracy
  let mapeStr = 'N/A';
  if (n >= 8) {
    const holdout = sales.slice(-4);
    const trainSales = sales.slice(0,-4), tn = trainSales.length;
    const txs = trainSales.map((_,i)=>i), txM=avg(txs), tyM=avg(trainSales);
    const tslope = txs.reduce((s,x,i)=>s+(x-txM)*(trainSales[i]-tyM),0)/txs.reduce((s,x)=>s+(x-txM)**2,0);
    const ticept = tyM-tslope*txM;
    let tema=trainSales[0]; trainSales.forEach(v=>{tema=alpha*v+(1-alpha)*tema;});
    const hPred = holdout.map((_,i)=>Math.max(0,0.6*(ticept+tslope*(tn+i))+0.4*tema));
    const mape  = holdout.reduce((s,a,i)=>s+Math.abs((a-hPred[i])/(a||1)),0)/holdout.length*100;
    const acc   = Math.max(0,100-mape);
    mapeStr = acc.toFixed(1)+'% '+(acc>=90?'🟢':acc>=75?'🟡':'🔴');
  }

  document.getElementById('forecastKpis').innerHTML = [
    {label:'Forecast Total',  val:fmtCur(forecastValues.reduce((a,b)=>a+b,0))},
    {label:'Peak Week',       val:fmtCur(Math.max(...forecastValues))},
    {label:'Avg Weekly',      val:fmtCur(avg(forecastValues))},
    {label:'Trend Direction', val:(trendPct>=0?'▲ ':'▼ ')+Math.abs(trendPct).toFixed(1)+'%'},
    {label:'Model Accuracy',  val:mapeStr},
  ].map(k=>`<div class="fkpi"><div class="fkpi-val">${k.val}</div><div class="fkpi-lab">${k.label}</div></div>`).join('');

  // If alert section exists already, update its forecast demand display
  updateAlertForecastDemand();
}

/* ========================
   YoY COMPARISON CHART
   ======================== */
function buildYoYChart() {
  const el = document.getElementById('yoyChart'); if (!el) return;
  const src = rawMappedData.length ? rawMappedData : mappedData;
  const byMonth = {};
  src.forEach(r=>{
    const y=r.date.getFullYear(), m=r.date.getMonth(), k=`${y}-${m}`;
    if(!byMonth[k]) byMonth[k]={y,m,s:0};
    byMonth[k].s+=r.sales;
  });
  const years=[...new Set(Object.values(byMonth).map(v=>v.y))].sort();
  if(years.length<2){el.innerHTML='<div style="padding:2rem;color:var(--text-3);font-size:0.82rem">Need 2+ years of data for YoY comparison.</div>';return;}
  const curY=years[years.length-1], prevY=years[years.length-2];
  const mons=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const cur=mons.map((_,m)=>byMonth[`${curY}-${m}`]?.s||0);
  const prev=mons.map((_,m)=>byMonth[`${prevY}-${m}`]?.s||0);
  Plotly.newPlot('yoyChart',[
    {x:mons,y:prev,type:'bar',name:String(prevY),marker:{color:'rgba(139,92,246,0.55)'},hovertemplate:`<b>%{x} ${prevY}</b><br>$%{y:,.0f}<extra></extra>`},
    {x:mons,y:cur, type:'bar',name:String(curY), marker:{color:'rgba(99,102,241,0.9)'},hovertemplate:`<b>%{x} ${curY}</b><br>$%{y:,.0f}<extra></extra>`},
  ],{...plotLayout(`${prevY} vs ${curY} — Year-over-Year`),height:280,barmode:'group',bargap:0.22},plotCfg());
}

/* ========================
   PRODUCT / SKU CHART
   ======================== */
function buildProductChart() {
  const sec = document.getElementById('productSection');
  const prodData = (rawMappedData.length?rawMappedData:mappedData).filter(r=>r.product);
  if(!prodData.length){if(sec)sec.style.display='none';return;}
  const pm={};
  prodData.forEach(r=>{pm[r.product]=(pm[r.product]||0)+r.sales;});
  const sorted=Object.entries(pm).sort((a,b)=>b[1]-a[1]).slice(0,12);
  if(sec)sec.style.display='';
  const el=document.getElementById('productChart');if(!el)return;
  const colors=['#6366f1','#8b5cf6','#06b6d4','#10b981','#f97316','#eab308'];
  Plotly.newPlot('productChart',[{
    type:'bar',orientation:'h',
    x:sorted.map(e=>e[1]),y:sorted.map(e=>e[0]),
    marker:{color:sorted.map((_,i)=>colors[i%colors.length])},
    hovertemplate:'<b>%{y}</b><br>$%{x:,.0f}<extra></extra>',
  }],{...plotLayout(''),height:Math.max(220,sorted.length*30+60),margin:{l:130,r:10,t:10,b:30}},plotCfg());
}

/* ========================
   PERSIST — IndexedDB (per-user)
   ======================== */

/* ========================
   ALERTS & NOTIFICATIONS
   ======================== */
function maybeAddAlertSection() {
  if (document.getElementById('alertSection')) return; // already injected
  if (!document.getElementById('resultsWrap') || document.getElementById('resultsWrap').style.display === 'none') return;
  const sessionEmail = getSession()?.email || '';

  // Build & inject alert section before the footer
  const footer = document.querySelector('.footer');
  const section = document.createElement('section');
  section.id        = 'alertSection';
  section.className = 'app-section';
  section.innerHTML = `
    <div class="container">
      <div class="section-label">Alerts &amp; Notifications</div>
      <h2 class="section-title">Inventory <span class="gradient-text">Alert Engine</span></h2>
      <p class="section-sub">Set your inventory level and safety threshold. Salevora automatically checks conditions after every analysis and fires instant browser alerts — no manual action needed.</p>

      <div class="alert-panel">
        <div class="alert-panel-title">⚙️ Configure Alert Parameters</div>
        <div class="alert-panel-sub">Alerts fire automatically when an alert condition is detected. You can also trigger a manual check anytime.</div>

        <div class="alert-grid">
          <div class="alert-field">
            <label>📦 Current Inventory (units)</label>
            <input type="number" id="alertInventory" value="10000" min="0" oninput="runAlertCheck()" />
          </div>
          <div class="alert-field">
            <label>⚠️ Safety Stock Threshold (units)</label>
            <input type="number" id="alertThreshold" value="15000" min="0" oninput="runAlertCheck()" />
          </div>
          <div class="alert-field">
            <label>🕐 Scheduled Auto-Check Time</label>
            <input type="time" id="alertTime" value="10:00" />
          </div>
        </div>

        <div id="alertForecastDisplay" style="font-size:0.83rem;color:var(--text-2);margin-bottom:0.5rem"></div>

        <div style="display:flex;gap:0.75rem;flex-wrap:wrap;margin-bottom:0;margin-top:0.5rem">
          <button class="btn btn-primary" onclick="runAlertCheck()">🔍 Check Now</button>
          <button class="btn btn-outline" onclick="scheduleAlert()">⏰ Schedule Daily Auto-Check</button>
        </div>

        <div id="alertResult" class="alert-result" style="display:none"></div>
        <div id="scheduleStatus" style="font-size:0.8rem;color:var(--text-3);margin-top:0.75rem"></div>
      </div>
    </div>`;

  footer.parentElement.insertBefore(section, footer);
  runAlertCheck();
  startAlertScheduler();
}

function getForecastedDemand() {
  if (!forecastValues.length) return 0;
  // next-4-week demand estimate
  return forecastValues.slice(0,4).reduce((a,b)=>a+b,0);
}

function updateAlertForecastDemand() {
  const el = document.getElementById('alertForecastDisplay');
  if (!el) return;
  const demand = getForecastedDemand();
  el.textContent = `📊 Forecasted demand (next 4 weeks): ${fmtCur(demand)} units based on your uploaded data.`;
}

function runAlertCheck(silent = false) {
  const inventory = parseFloat(document.getElementById('alertInventory')?.value) || 0;
  const threshold = parseFloat(document.getElementById('alertThreshold')?.value) || 0;
  const demand    = getForecastedDemand();
  const resultEl  = document.getElementById('alertResult');

  updateAlertForecastDemand();
  if (!resultEl) return;
  resultEl.style.display = '';

  const isLow       = inventory < threshold;
  const isHighDemand = demand > inventory;

  let cls, title, body;
  if (isLow && isHighDemand) {
    cls   = 'danger';
    title = '🚨 CRITICAL: Inventory Alert Triggered!';
    body  = `Both conditions are met — inventory is below safety stock AND forecasted demand exceeds available stock.`;
    if (!silent) {
      const inv = inventory.toLocaleString();
      addNavNotification(
        '🚨 Critical: Inventory Alert',
        `Stock (${inv} units) is critically low — forecasted demand ${fmtCur(demand)} exceeds it.`,
        'danger'
      );
      toast('🚨 Critical inventory alert! Stock is critically low.', 'warning', 6000);
    }
  } else if (isLow) {
    cls   = 'warning';
    title = '⚠️ WARNING: Inventory Below Safety Threshold';
    body  = `Inventory (${inventory.toLocaleString()}) is below your threshold (${threshold.toLocaleString()}), but demand is within safe range for now.`;
    if (!silent) {
      addNavNotification(
        '⚠️ Warning: Low Inventory',
        `Stock (${inventory.toLocaleString()}) is below safety threshold (${threshold.toLocaleString()}).`,
        'warning'
      );
      toast('⚠️ Inventory below safety threshold!', 'warning', 5000);
    }
  } else {
    cls   = 'safe';
    title = '✅ Inventory Levels Are Stable';
    body  = `Inventory (${inventory.toLocaleString()}) is above threshold (${threshold.toLocaleString()}) and demand is fully covered.`;
  }

  resultEl.className = 'alert-result ' + cls;
  resultEl.innerHTML = `
    <div class="alert-result-title">${title}</div>
    <div>${body}</div>
    <div class="alert-stats">
      <div class="alert-stat"><span class="alert-stat-val">${inventory.toLocaleString()}</span><span class="alert-stat-lab">Current Stock</span></div>
      <div class="alert-stat"><span class="alert-stat-val">${threshold.toLocaleString()}</span><span class="alert-stat-lab">Safety Threshold</span></div>
      <div class="alert-stat"><span class="alert-stat-val">${fmtCur(demand)}</span><span class="alert-stat-lab">Forecasted Demand (4 wk)</span></div>
    </div>`;
}

/* Browser notification permission - kept for legacy, but navbar bell is primary */
function requestBrowserNotif() {
  toast('🔔 Notifications appear in the navbar bell above.', 'info');
}

/* Scheduled daily auto-check */
let alertIntervalId = null;
function scheduleAlert() {
  const timeVal  = document.getElementById('alertTime')?.value || '10:00';
  const [h, m]   = timeVal.split(':').map(Number);
  const statusEl = document.getElementById('scheduleStatus');
  if (statusEl) statusEl.textContent = `⏰ Daily auto-check scheduled for ${fmtTime12(h,m)} — will fire automatically. Keep this tab open.`;
  toast(`✅ Auto-check scheduled for ${fmtTime12(h,m)} every day.`, 'success');

  if (alertIntervalId) clearInterval(alertIntervalId);
  alertIntervalId = setInterval(() => {
    const now = new Date();
    if (now.getHours() === h && now.getMinutes() === m) {
      runAlertCheck(); // fires toast + browser notification automatically inside
    }
  }, 60000);
}

function startAlertScheduler() {
  // No-op: scheduler starts when user explicitly sets a time
}

function fmtTime12(h, m) {
  const ampm = h >= 12 ? 'PM' : 'AM';
  const h12  = h % 12 || 12;
  return `${h12}:${String(m).padStart(2,'0')} ${ampm}`;
}

/* ========================
   TABLE
   ======================== */
function buildTable() {
  filteredRows = [...mappedData];
  currentPage  = 1;
  renderTable();
}

function filterTable() {
  const q = (document.getElementById('tableSearch').value || '').toLowerCase();
  filteredRows = q
    ? mappedData.filter(r => fmtDate(r.date).includes(q) || String(r.sales).includes(q) || r.category.toLowerCase().includes(q))
    : [...mappedData];
  currentPage = 1;
  renderTable();
}

function renderTable() {
  const head   = document.getElementById('tableHead');
  const body   = document.getElementById('tableBody');
  const meta   = document.getElementById('tableMeta');
  const cols   = ['date','sales','revenue','category'];
  head.innerHTML = '<tr>'+cols.map(c=>`<th>${c.toUpperCase()}</th>`).join('')+'</tr>';

  const total = filteredRows.length;
  const start = (currentPage-1)*PAGE_SIZE;
  const shown = filteredRows.slice(start, start+PAGE_SIZE);

  body.innerHTML = shown.length
    ? shown.map(r=>`<tr><td>${fmtDate(r.date)}</td><td>${fmtCur(r.sales)}</td><td>${fmtCur(r.revenue)}</td><td>${r.category}</td></tr>`).join('')
    : '<tr><td colspan="4" style="text-align:center;padding:2rem;color:var(--text-3)">No matching records.</td></tr>';

  meta.textContent = `Showing ${start+1}–${Math.min(start+PAGE_SIZE,total)} of ${total.toLocaleString()} records`;

  const pages = Math.ceil(total/PAGE_SIZE);
  const pgWrap = document.getElementById('pagination');
  if (pages <= 1) { pgWrap.innerHTML = ''; return; }
  const show = [];
  if (currentPage > 2) show.push(1,'…');
  for (let i=Math.max(1,currentPage-1); i<=Math.min(pages,currentPage+1); i++) show.push(i);
  if (currentPage < pages-1) show.push('…',pages);
  pgWrap.innerHTML = show.map(p=>p==='…'
    ? `<span style="color:var(--text-3);padding:0 0.3rem">…</span>`
    : `<button class="page-btn ${p===currentPage?'active':''}" onclick="goPage(${p})">${p}</button>`).join('');
}

function goPage(p) { currentPage=p; renderTable(); document.getElementById('tableSection').scrollIntoView({behavior:'smooth',block:'start'}); }

function downloadCSV() {
  const rows = filteredRows.map(r=>[fmtDate(r.date),r.sales.toFixed(2),r.revenue.toFixed(2),r.category].join(','));
  const blob = new Blob([['date,sales,revenue,category',...rows].join('\n')],{type:'text/csv'});
  const a = document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='salevora_export.csv'; a.click();
}

/* ========================
   PLOTLY HELPERS
   ======================== */
function plotLayout(title) {
  return {
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Inter,sans-serif',color:'#94a3b8',size:12},
    title: title ? {text:title,font:{size:13,color:'#cbd5e1'},x:0.01} : undefined,
    margin:{l:50,r:20,t:title?42:16,b:40},
    xaxis:{gridcolor:'rgba(148,163,184,0.08)',zerolinecolor:'rgba(148,163,184,0.1)',tickfont:{color:'#64748b',size:11}},
    yaxis:{gridcolor:'rgba(148,163,184,0.08)',zerolinecolor:'rgba(148,163,184,0.1)',tickfont:{color:'#64748b',size:11}},
    legend:{bgcolor:'rgba(13,20,39,0.7)',bordercolor:'rgba(99,102,241,0.2)',borderwidth:1,font:{color:'#cbd5e1',size:11}},
  };
}
function plotCfg() { return {displayModeBar:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d'],responsive:true}; }

/* ========================
   HELPERS
   ======================== */
function avg(arr)      { return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0; }
function fmtCur(v)     { return '$'+Number(v).toLocaleString('en-US',{maximumFractionDigits:0}); }
function fmtDate(d)    { return d instanceof Date ? d.toISOString().slice(0,10) : String(d); }
