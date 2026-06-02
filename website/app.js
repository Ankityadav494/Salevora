/* =============================================
   SALEVORA — app.js
   Auth + File Processing + Predictions + Alerts
   Connected to Python FastAPI backend
   ============================================= */

let backendOnline = false;
let lastBackendForecast = null;
let currentFileName  = '';

/* ========================
   AUTH — UI Helpers
   ======================== */
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

function showTab(tab) {
  document.getElementById('loginForm').style.display    = tab === 'login'    ? '' : 'none';
  document.getElementById('registerForm').style.display = tab === 'register' ? '' : 'none';
  document.getElementById('otpForm').style.display        = 'none';
  document.getElementById('forgotForm').style.display     = 'none';
  document.getElementById('resetForm').style.display      = 'none';
  document.getElementById('tabLogin').classList.toggle('active',    tab === 'login');
  document.getElementById('tabRegister').classList.toggle('active', tab === 'register');
  clearAuthMessages();
  if (tab === 'login') clearPendingReset();
}

function showForgotForm() {
  document.getElementById('loginForm').style.display = 'none';
  document.getElementById('registerForm').style.display = 'none';
  document.getElementById('otpForm').style.display = 'none';
  document.getElementById('resetForm').style.display = 'none';
  document.getElementById('forgotForm').style.display = '';
  const loginEmail = document.getElementById('loginEmail')?.value?.trim();
  if (loginEmail) document.getElementById('forgotEmail').value = loginEmail;
  clearAuthMessages();
}

function showResetForm(email) {
  document.getElementById('loginForm').style.display = 'none';
  document.getElementById('registerForm').style.display = 'none';
  document.getElementById('otpForm').style.display = 'none';
  document.getElementById('forgotForm').style.display = 'none';
  document.getElementById('resetForm').style.display = '';
  document.getElementById('resetSub').textContent =
    `Enter the 6-digit code we sent to ${email} and choose a new password.`;
  document.getElementById('resetCode').value = '';
  document.getElementById('resetPassword').value = '';
  const err = document.getElementById('resetError');
  if (err) { err.style.display = 'none'; err.textContent = ''; }
}

function showOtpForm(email, purpose) {
  document.getElementById('loginForm').style.display = 'none';
  document.getElementById('registerForm').style.display = 'none';
  document.getElementById('otpForm').style.display = '';
  document.getElementById('otpSub').textContent =
    `Enter the 6-digit code we sent to ${email}`;
  document.getElementById('otpCode').value = '';
  const err = document.getElementById('otpError');
  if (err) { err.style.display = 'none'; err.textContent = ''; }
}

function cancelOtp() {
  const purpose = getPendingOtp()?.purpose || 'login';
  clearPendingOtp();
  showTab(purpose === 'register' ? 'register' : 'login');
}

function clearAuthMessages() {
  ['loginError','regError','regSuccess','otpError','forgotError','forgotSuccess','resetError'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.style.display = 'none'; el.textContent = ''; }
  });
}

function showError(id, msg)   { const el = document.getElementById(id); el.textContent = msg; el.style.display = ''; }
function showSuccess(id, msg) { const el = document.getElementById(id); el.textContent = msg; el.style.display = ''; }

function togglePwd(inputId, btn) {
  const input = document.getElementById(inputId);
  if (input.type === 'password') { input.type = 'text'; btn.textContent = 'Hide'; }
  else                           { input.type = 'password'; btn.textContent = 'Show'; }
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
  try {
    const user = await loginUser(email, password);
    try {
      enterApp(user);
    } catch (uiErr) {
      console.error('enterApp failed:', uiErr);
      showError('loginError', '❌ Signed in, but the dashboard failed to load. Hard refresh (Ctrl+Shift+R) and try again.');
    }
  } catch (err) {
    showError('loginError', `❌ ${friendlyError(err.message) || 'That email or password is not correct.'}`);
  }
  btn.textContent = 'Sign in'; btn.disabled = false;
}

async function handleForgotPassword(e) {
  e.preventDefault();
  clearAuthMessages();
  const email = document.getElementById('forgotEmail').value.trim().toLowerCase();
  const btn = document.getElementById('forgotSubmit');
  btn.textContent = 'Sending…'; btn.disabled = true;
  try {
    const res = await requestPasswordReset(email);
    showResetForm(email);
    toast(res.message || 'Check your email for a reset code.', 'success');
  } catch (err) {
    showError('forgotError', `❌ ${friendlyError(err.message)}`);
  }
  btn.textContent = 'Send reset code'; btn.disabled = false;
}

async function handleResetPassword(e) {
  e.preventDefault();
  clearAuthMessages();
  const pending = getPendingReset();
  if (!pending?.email) { showTab('login'); return; }
  const otp = document.getElementById('resetCode').value.trim();
  const password = document.getElementById('resetPassword').value;
  const btn = document.getElementById('resetSubmit');
  if (password.length < 6) {
    showError('resetError', '❌ Password must be at least 6 characters.');
    return;
  }
  btn.textContent = 'Updating…'; btn.disabled = true;
  try {
    const user = await resetPassword(pending.email, otp, password);
    enterApp(user);
    toast('Password updated. Welcome back!', 'success');
  } catch (err) {
    showError('resetError', `❌ ${friendlyError(err.message)}`);
  }
  btn.textContent = 'Update password'; btn.disabled = false;
}

async function resendResetCode() {
  const pending = getPendingReset();
  if (!pending?.email) { showForgotForm(); return; }
  try {
    const res = await requestPasswordReset(pending.email);
    toast(res.message || 'Reset code resent.', 'success');
  } catch (err) {
    showError('resetError', `❌ ${friendlyError(err.message)}`);
  }
}

async function handleOtpVerify(e) {
  e.preventDefault();
  const pending = getPendingOtp();
  if (!pending) { showTab('login'); return; }
  const otp = document.getElementById('otpCode').value.trim();
  const btn = document.getElementById('otpSubmit');
  btn.textContent = 'Verifying…'; btn.disabled = true;
  try {
    const user = await verifyOtpAndLogin(pending.email, otp, pending.purpose);
    enterApp(user);
  } catch (err) {
    showError('otpError', `❌ ${friendlyError(err.message)}`);
  }
  btn.textContent = 'Verify & Continue'; btn.disabled = false;
}

async function resendOtp() {
  const pending = getPendingOtp();
  if (!pending) return;
  try {
    const res = await requestOtp(pending.email, pending.purpose, {
      password: pending.password,
      name: pending.name,
    });
    toast(res.message || 'Code resent.', 'success');
  } catch (err) {
    showError('otpError', `❌ ${friendlyError(err.message)}`);
  }
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
  const btn      = document.getElementById('regSubmit');
  if (password.length < 6)  { showError('regError', '❌ Password must be at least 6 characters.'); return; }
  btn.textContent = 'Creating account…'; btn.disabled = true;
  try {
    const otpRes = await requestOtp(email, 'register', { password, name });
    if (otpRes.skip_otp) {
      const user = await registerUser(name, email, password);
      enterApp(user);
    } else {
      showOtpForm(email, 'register');
      showSuccess('regSuccess', 'Check your email for a sign-in code.');
    }
  } catch (err) {
    showError('regError', `❌ ${friendlyError(err.message)}`);
  }
  btn.textContent = 'Create Account'; btn.disabled = false;
}

/* ========================
   APP — Enter / Exit
   ======================== */
function enterApp(user) {
  document.body.classList.remove('page-auth');
  document.body.classList.add('page-app', 'page-dashboard');
  document.getElementById('authScreen').style.display = 'none';
  document.getElementById('appScreen').style.display  = 'flex';
  setSidebarUser(user);
  checkRestoreData();
  bootstrapApp();
  if (typeof decorateIcons === 'function') decorateIcons();
  if (typeof decorateSidebar === 'function') decorateSidebar();
  if (typeof initAppShellAnimation === 'function') initAppShellAnimation(document.getElementById('appScreen'));
  setSidebarUser(user);
  if (typeof refreshReveals === 'function') refreshReveals();
  if (typeof initReveal === 'function') initReveal();
  toast(`Welcome back, ${user.name.split(' ')[0]}!`, 'success');
}

async function bootstrapApp() {
  if (!getToken()) return;
  try {
    await checkBackend();
    const status = await SalevoraAPI.appStatus();
    if (status.sales?.rows) {
      const key = 'sv_server_dismiss_' + (getSession()?.email || '');
      if (!localStorage.getItem(key)) {
        const banner = document.getElementById('serverDataBanner');
        const info = document.getElementById('serverDataInfo');
        if (banner && info) {
          info.textContent = ` ${status.sales.rows.toLocaleString()} rows · ${status.sales.date_start} to ${status.sales.date_end}`;
          banner.style.display = '';
        }
      }
    }
    if (status.inventory?.alerts > 0) {
      addNavNotification(
        'Stock reminders',
        `${status.inventory.alerts} product(s) may need attention — open Stock Levels.`,
        status.inventory.critical ? 'danger' : 'warning'
      );
    }
  } catch (e) {
    console.warn('Bootstrap failed:', e);
  }
}

function dismissServerBanner() {
  localStorage.setItem('sv_server_dismiss_' + (getSession()?.email || ''), '1');
  const b = document.getElementById('serverDataBanner');
  if (b) b.style.display = 'none';
}

async function loadServerSales() {
  try {
    setUploadStatus('info', 'Loading your saved sales…');
    const res = await SalevoraAPI.downloadData();
    if (!res.data?.length) { toast('No saved sales found.', 'warning'); return; }
    rawData = res.data;
    currentFileName = 'saved_sales.csv';
    allColumns = Object.keys(res.data[0]);
    const lc = allColumns.map(c => c.toLowerCase());
    colMap.date     = allColumns.find((_, i) => lc[i].includes('date')) || 'date';
    colMap.sales    = allColumns.find((_, i) => lc[i].includes('sales')) || 'sales';
    colMap.revenue  = allColumns.find((_, i) => lc[i].includes('revenue')) || '';
    colMap.category = allColumns.find((_, i) => lc[i].includes('category')) || '';
    dismissServerBanner();
    runPrediction();
    toast(`Loaded ${res.rows.toLocaleString()} saved sales rows.`, 'success');
  } catch (e) {
    toast('Could not load saved sales.', 'warning');
  }
}

function handleLogout() {
  clearSession();
  document.body.classList.add('page-auth');
  document.body.classList.remove('page-app', 'page-dashboard');
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

function onDashSearch(q) {
  const tableSearch = document.getElementById('tableSearch');
  if (tableSearch) {
    tableSearch.value = q;
    if (typeof filterTable === 'function') filterTable();
  }
  if (q && document.getElementById('resultsWrap')?.style.display !== 'none') {
    document.getElementById('tableSection')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function pctRingSvg(pct, color = 'var(--accent)') {
  const p = Math.min(100, Math.max(0, Math.abs(pct)));
  const r = 42;
  const circ = 2 * Math.PI * r;
  const dash = circ * (p / 100);
  return `<svg class="ring-chart" viewBox="0 0 100 100" aria-hidden="true">
    <circle cx="50" cy="50" r="${r}" fill="none" stroke="var(--border)" stroke-width="9"/>
    <circle cx="50" cy="50" r="${r}" fill="none" stroke="${color}" stroke-width="9"
      stroke-dasharray="${dash} ${circ}" stroke-linecap="round" transform="rotate(-90 50 50)"/>
  </svg>`;
}

const RING_COLORS = { revenue: '#2F5233', momentum: '#4A7A50', growth: '#7BA882' };

function ringStatCard(label, pct, display, sub, variant = 'revenue') {
  const barW = Math.min(100, Math.max(8, Math.abs(pct)));
  const color = RING_COLORS[variant] || RING_COLORS.revenue;
  return `<div class="stat-ring-card stat-ring-card--${variant}">
    <div class="stat-ring-label">${label}</div>
    <div class="stat-ring-wrap">
      ${pctRingSvg(pct, color)}
      <div class="stat-ring-pct">${display}</div>
    </div>
    <div class="stat-ring-sub">${sub}</div>
    <div class="stat-ring-bar"><div class="stat-ring-bar-fill" style="width:${barW}%"></div></div>
  </div>`;
}

// Auto-login if session active
window.addEventListener('DOMContentLoaded', async () => {
  checkBackend();
  const user = await validateSession();
  if (user) enterApp(user);
});

async function checkBackend() {
  try {
    await SalevoraAPI.health();
    backendOnline = true;
  } catch {
    backendOnline = false;
  }
}

async function uploadToBackend(file, mode = 'replace') {
  if (!backendOnline) await checkBackend();
  if (!backendOnline) return null;
  try {
    return await SalevoraAPI.uploadFile(file, mode);
  } catch (e) {
    console.warn('Backend upload failed:', e);
    toast('Could not save to the server — showing results on this page only.', 'warning');
    return null;
  }
}

async function fetchBackendForecast(horizon, model) {
  if (!backendOnline) await checkBackend();
  if (!backendOnline) return null;
  try {
    return await SalevoraAPI.forecast(horizon, model);
  } catch {
    return null;
  }
}

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
      mappedData: mappedData.slice(0, 5000).map(r => ({ d: r.date.getTime(), s: r.sales, rv: r.revenue, c: r.category }))
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
    mappedData = saved.mappedData.map(r => ({ date: new Date(r.d), sales: r.s, revenue: r.rv, category: r.c }));
    weeklyData = saved.weeklyData.map(w => ({ date: new Date(w.d), sales: w.s, revenue: w.r }));
    colMap        = saved.colMap;
    currentFileName = saved.fileName;
    document.getElementById('restoreBanner').style.display = 'none';
    document.getElementById('resultsWrap').style.display = '';
    document.getElementById('colConfig').style.display = 'none';
    buildKPIs(); buildTrendChart(); buildCategoryChart(); buildMonthlyChart();
    buildTopPerformers(); updateForecast(); buildTable();
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
  const statusTxt = !goal ? 'Set a target to track progress' : pct >= 80 ? 'On track' : pct >= 50 ? 'At risk' : 'Behind target';
  const now = new Date();
  const daysLeft = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate() - now.getDate();
  el.style.display = '';
  el.innerHTML = `
    <div class="goal-card">
      <div class="goal-header">
        <div>
          <div class="goal-title">Monthly sales goal</div>
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
  body{font-family:'DM Sans',Arial,sans-serif;color:#1F1C18;background:#FAF8F4;font-size:13px}
  .header{background:#2F5233;color:#fff;padding:28px 40px;display:flex;justify-content:space-between;align-items:center}
  .logo{font-family:Georgia,serif;font-size:26px;font-weight:400;letter-spacing:-0.02em}
  .header-sub{font-size:11px;opacity:.85;margin-top:5px}
  .header-right{text-align:right;font-size:11px;opacity:.85;line-height:1.7}
  .body{padding:30px 40px}
  .insight{background:#E6EFE7;border-left:4px solid #2F5233;padding:11px 15px;margin-bottom:20px;border-radius:4px;font-size:12px;color:#243F27;font-weight:500}
  .summary{font-size:11px;color:#5C5650;margin-bottom:22px}
  h2{font-size:14px;font-weight:700;color:#1F1C18;border-bottom:2px solid #2F5233;padding-bottom:5px;margin-bottom:12px;margin-top:24px}
  .kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:8px}
  .kpi{background:#fff;border:1px solid #DDD8CE;border-top:3px solid #2F5233;border-radius:6px;padding:11px}
  .kpi-l{font-size:9px;text-transform:uppercase;letter-spacing:.07em;color:#8A837A;margin-bottom:4px}
  .kpi-v{font-size:17px;font-weight:800;color:#1F1C18}
  .kpi-t{font-size:10px;color:#5C5650;margin-top:3px}
  .chart{width:100%;height:auto;border-radius:6px;border:1px solid #DDD8CE;margin-bottom:6px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  thead th{background:#2F5233;color:#fff;padding:8px 10px;text-align:left;font-size:11px}
  tbody td{padding:8px 10px;border-bottom:1px solid #F3F0EA}
  tbody tr:nth-child(even) td{background:#FAF8F4}
  .footer{margin-top:30px;padding-top:12px;border-top:1px solid #DDD8CE;display:flex;justify-content:space-between;font-size:10px;color:#8A837A}
  @media print{
    body{-webkit-print-color-adjust:exact;print-color-adjust:exact}
    h2{page-break-after:avoid}
    .no-break{page-break-inside:avoid}
  }
</style></head><body>
<div class="header">
  <div><div class="logo">Salevora</div><div class="header-sub">Sales report</div></div>
  <div class="header-right"><strong>${today}</strong><br>Generated by Salevora</div>
</div>
<div class="body">
  ${insight?`<div class="insight">${insight}</div>`:''}
  ${summary?`<p class="summary">${summary}</p>`:''}
  <h2>Key Performance Indicators</h2>
  <div class="kpi-grid no-break">${kpiCards}</div>
  ${trendImg?`<h2>Historical Sales Trend + Anomaly Detection</h2><img class="chart no-break" src="${trendImg}" />`:''}
  ${forecastImg?`<h2>What sales may look like next</h2><img class="chart no-break" src="${forecastImg}" />`:''}
  ${perfRows?`<h2>Top Performers by Category</h2><table class="no-break"><thead><tr><th>Rank</th><th>Category</th><th>Stats</th></tr></thead><tbody>${perfRows}</tbody></table>`:''}
  <div class="footer">
    <span>Salevora</span>
    <span>Your data stays on your account</span>
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
  if (list) list.innerHTML = '<div class="notif-empty">Nothing new</div>';
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
let colMap = { date:'', sales:'', revenue:'', category:'' };
let filteredRows = [], currentPage = 1;
const PAGE_SIZE = 25;

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
      error: () => setUploadStatus('error', 'Could not read this file. Please check the format and try again.'),
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
    setUploadStatus('error', 'Please use an Excel or spreadsheet file (.xlsx, .xls, or .csv).');
  }
}

function setUploadStatus(type, msg) {
  const el = document.getElementById('uploadStatus');
  el.className = 'upload-status ' + type;
  el.textContent = msg;
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
    return { date: d, sales, revenue, category };
  }).filter(Boolean).sort((a,b) => a.date - b.date);

  if (!mappedData.length) {
    setUploadStatus('error', 'No valid date rows found. Check your date column (use YYYY-MM-DD).'); return;
  }

  weeklyData = aggregateWeekly(mappedData);

  document.getElementById('resultsWrap').style.display = '';
  buildKPIs();
  buildTrendChart();
  buildCategoryChart();
  buildMonthlyChart();
  buildTopPerformers();
  updateForecast();
  buildTable();
  if (typeof refreshReveals === 'function') refreshReveals();
  if (typeof animateStatRings === 'function') animateStatRings();

  document.getElementById('kpiSection').scrollIntoView({ behavior:'smooth', block:'start' });
  toast('Done! Scroll down to see your sales outlook.', 'success');
  saveToStorage();

  if (currentFileName && fileInput.files[0]) {
    uploadToBackend(fileInput.files[0]).then(res => {
      if (res) {
        const inv = res.inventory;
        const invMsg = inv
          ? ` · Stock check: ${inv.skus_from_sales} products, ${inv.alert_count} need attention`
          : '';
        toast(`✅ Saved (${res.rows_saved?.toLocaleString()} rows)${invMsg}`, 'success');
        notifyAlertEmail(res.alert_email);
        if (inv?.alert_count > 0) {
          toast('Some products may run low — check Stock Levels.', 'warning', 5000);
        }
        updateForecast();
      }
    });
  }
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
  if(ib){ib.style.display=parts.length?'':'none'; ib.innerHTML=parts.join(' · ')+'.';}

  document.getElementById('summaryText').textContent =
    `${rows.toLocaleString()} records · ${fmtDate(start)} → ${fmtDate(end)} · ${weeklyData.length} weeks · ${cats.length} categories`;

  const momentumPct = Math.min(100, Math.abs(delta));
  const growthPct = mom != null ? Math.min(100, Math.abs(mom)) : (yoy != null ? Math.min(100, Math.abs(yoy)) : 50);
  const revenueShare = total > 0 ? Math.min(100, Math.round((last4 / (sales || 1)) * 100 * 4)) : 0;

  const ringEl = document.getElementById('ringStats');
  if (ringEl) {
    ringEl.innerHTML = [
      ringStatCard('Revenue', revenueShare || 25, fmtCur(total), 'Total revenue to date', 'revenue'),
      ringStatCard('Momentum', momentumPct || 0, `${delta >= 0 ? '+' : ''}${delta.toFixed(0)}%`, 'Last 4 weeks vs prior 4', 'momentum'),
      ringStatCard('Growth', growthPct, mom != null ? `${mom >= 0 ? '+' : ''}${mom.toFixed(0)}%` : (yoy != null ? `${yoy >= 0 ? '+' : ''}${yoy.toFixed(0)}%` : '—'), mom != null ? 'Month over month' : 'Year over year', 'growth'),
    ].join('');
  }

  const statBoxes = document.getElementById('statBoxes');
  if (statBoxes) {
    statBoxes.innerHTML = [
      { lab: 'Total sales', val: fmtCur(sales) },
      { lab: 'Daily avg', val: fmtCur(sales / (days || 1)) },
      { lab: 'Categories', val: String(cats.length) },
    ].map(b => `<div class="dash-stat-box"><div class="dash-stat-box-val">${b.val}</div><div class="dash-stat-box-lab">${b.lab}</div></div>`).join('');
  }

  document.getElementById('kpiGrid').innerHTML = [
    { label:'Total revenue',  value:fmtCur(total),         sub:`${rows.toLocaleString()} records` },
    { label:'Last 4 weeks',   value:fmtCur(last4),         trend:delta },
    mom!=null&&{ label:'Month over month', value:fmtCur(last30), trend:mom },
    yoy!=null&&{ label:'Year over year',   value:fmtCur(last365), trend:yoy },
    bestCat  &&{ label:'Top category', value:bestCat[0], sub:fmtCur(bestCat[1])+' in sales' },
    { label:'Period start',  value:fmtDate(start), sub:'Earliest record' },
    { label:'Period end',    value:fmtDate(end),   sub:'Latest record' },
  ].filter(Boolean).map((k, i)=>`
    <div class="kpi-card" style="--reveal-delay:${Math.min(i * 0.05, 0.35)}s">
      <div class="kpi-card-top">
        <span class="kpi-icon">${typeof svIcon === 'function' ? svIcon(kpiIconFor(k.label), 20) : ''}</span>
        <div class="kpi-label">${k.label}</div>
      </div>
      <div class="kpi-value">${k.value}</div>
      ${k.trend!=null
        ?`<div class="${k.trend>=0?'kpi-trend-up':'kpi-trend-down'}">${k.trend>=0?'↑':'↓'} ${Math.abs(k.trend).toFixed(1)}% vs prior</div>`
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
  if(anomDates.length) addNavNotification('Unusual sales week', `${anomDates.length} week(s) outside the normal range.`, 'warning');
  Plotly.newPlot('trendChart',[
    { x:dates,y:sales,type:'scatter',mode:'lines',name:'Weekly sales',
      line:{color:'#2F5233',width:2.2,shape:'spline'},fill:'tozeroy',fillcolor:'rgba(47,82,51,0.08)',
      hovertemplate:'<b>%{x|%b %d, %Y}</b><br>$%{y:,.0f}<extra></extra>' },
    { x:dates,y:ma4,type:'scatter',mode:'lines',name:'4-week average',
      line:{color:'#5A7A5E',width:1.8,dash:'dot'},hovertemplate:'<b>%{x|%b %d, %Y}</b><br>MA: $%{y:,.0f}<extra></extra>' },
    anomDates.length?{ x:anomDates,y:anomVals,type:'scatter',mode:'markers',name:'Unusual week',
      marker:{size:10,color:'#B91C1C',symbol:'circle',line:{width:2,color:'#fff'}},
      hovertemplate:'<b>%{x|%b %d, %Y}</b><br>$%{y:,.0f}<extra></extra>' }:null,
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
  const colors = ['#2F5233','#5A7A5E','#0369A1','#A16207','#4A674D','#C2410C'];
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
    marker:{color:vals.map(v=>v>=mean?'rgba(47,82,51,0.85)':'rgba(90,122,94,0.45)')},
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
  const colors=['#2F5233','#5A7A5E','#0369A1','#A16207','#4A674D'];
  const section=document.createElement('section');
  section.id='topPerfSection'; section.className='app-section';
  section.innerHTML=`
    <div class="container">
      <div class="section-label">Performance</div>
      <h2 class="section-title">Top <span class="accent-word">performers</span></h2>
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
  const weeks  = parseInt(document.getElementById('horizonSelect').value);
  const model  = document.getElementById('modelSelect')?.value || 'ensemble';

  fetchBackendForecast(weeks, model).then(data => {
    if (data?.forecast?.length) {
      lastBackendForecast = data;
      renderBackendForecast(data);
    } else {
      renderClientForecast(weeks);
    }
  });
}

function renderBackendForecast(data) {
  const sales = weeklyData.map(w => w.sales);
  const fDates = data.forecast.map(f => new Date(f.date));
  forecastValues = data.forecast.map(f => f.sales);
  const lower = data.forecast.map(f => f.lower);
  const upper = data.forecast.map(f => f.upper);

  const modelLabel = {
    ensemble: 'Smart mix',
    arima: 'Trend-based',
    prophet: 'Seasonal',
    lstm: 'Advanced patterns',
  }[data.model] || 'Estimate';

  Plotly.newPlot('forecastChart',[
    { x:weeklyData.map(w=>w.date),y:sales,type:'scatter',mode:'lines',name:'Past sales',
      line:{color:'#2F5233',width:2.2,shape:'spline'},fill:'tozeroy',fillcolor:'rgba(47,82,51,0.06)',
      hovertemplate:'<b>%{x|%b %d, %Y}</b><br>$%{y:,.0f}<extra></extra>' },
    { x:[...fDates,...fDates.slice().reverse()],y:[...upper,...lower.slice().reverse()],
      fill:'toself',fillcolor:'rgba(194,65,12,0.08)',line:{color:'transparent'},hoverinfo:'skip',name:'Range' },
    { x:fDates,y:forecastValues,type:'scatter',mode:'lines+markers',name:`Estimate (${modelLabel})`,
      line:{color:'#C2410C',width:2.2,dash:'dot'},marker:{size:6,color:'#C2410C',line:{width:1.5,color:'#fff'}},
      hovertemplate:'<b>%{x|%b %d, %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>' },
    { x:[weeklyData[weeklyData.length-1].date,fDates[0]],y:[sales[sales.length-1],forecastValues[0]],
      mode:'lines',line:{color:'#C2410C',width:1,dash:'dot'},showlegend:false,hoverinfo:'skip' },
  ],{ ...plotLayout(`Sales outlook — ${modelLabel}`), height:370 },plotCfg());

  const trendPct = forecastValues[0] ? ((forecastValues[forecastValues.length-1]-forecastValues[0])/forecastValues[0]*100) : 0;
  const acc = data.metrics_rolling?.accuracy ?? data.metrics?.accuracy;
  const mapeStr = acc != null ? `${acc.toFixed(1)}%` : '—';

  document.getElementById('forecastKpis').innerHTML = [
    {label:'Total expected',  val:fmtCur(forecastValues.reduce((a,b)=>a+b,0))},
    {label:'Best week',       val:fmtCur(Math.max(...forecastValues))},
    {label:'Average week',    val:fmtCur(avg(forecastValues))},
    {label:'Trend',           val:(trendPct>=0?'▲ ':'▼ ')+Math.abs(trendPct).toFixed(1)+'%'},
    {label:'How close we were', val:mapeStr},
    {label:'Method used',     val:modelLabel},
  ].map(k=>`<div class="fkpi"><div class="fkpi-val">${k.val}</div><div class="fkpi-lab">${k.label}</div></div>`).join('');
}

function renderClientForecast(weeks) {
  const smooth  = document.getElementById('smoothSelect')?.value || 'medium';
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
      line:{color:'#2F5233',width:2.2,shape:'spline'},fill:'tozeroy',fillcolor:'rgba(47,82,51,0.06)',
      hovertemplate:'<b>%{x|%b %d, %Y}</b><br>$%{y:,.0f}<extra></extra>' },
    { x:[...fDates,...fDates.slice().reverse()],y:[...upper,...lower.slice().reverse()],
      fill:'toself',fillcolor:'rgba(194,65,12,0.08)',line:{color:'transparent'},hoverinfo:'skip',name:'Range' },
    { x:fDates,y:forecastValues,type:'scatter',mode:'lines+markers',name:'Estimate',
      line:{color:'#C2410C',width:2.2,dash:'dot'},marker:{size:6,color:'#C2410C',line:{width:1.5,color:'#fff'}},
      hovertemplate:'<b>%{x|%b %d, %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>' },
    { x:[weeklyData[weeklyData.length-1].date,fDates[0]],y:[sales[sales.length-1],forecastValues[0]],
      mode:'lines',line:{color:'#C2410C',width:1,dash:'dot'},showlegend:false,hoverinfo:'skip' },
  ],{ ...plotLayout('Sales outlook (estimate)'), height:370 },plotCfg());

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
    mapeStr = acc.toFixed(1)+'%';
  }

  document.getElementById('forecastKpis').innerHTML = [
    {label:'Total expected',  val:fmtCur(forecastValues.reduce((a,b)=>a+b,0))},
    {label:'Best week',       val:fmtCur(Math.max(...forecastValues))},
    {label:'Average week',    val:fmtCur(avg(forecastValues))},
    {label:'Trend',           val:(trendPct>=0?'▲ ':'▼ ')+Math.abs(trendPct).toFixed(1)+'%'},
    {label:'How close we were', val:mapeStr},
  ].map(k=>`<div class="fkpi"><div class="fkpi-val">${k.val}</div><div class="fkpi-lab">${k.label}</div></div>`).join('');
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
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'DM Sans, sans-serif', color: '#5C5650', size: 12 },
    title: title ? { text: title, font: { size: 13, color: '#1F1C18', family: 'DM Sans' }, x: 0.01 } : undefined,
    margin: { l: 50, r: 20, t: title ? 42 : 16, b: 40 },
    xaxis: { gridcolor: 'rgba(31,28,24,0.06)', zerolinecolor: 'rgba(31,28,24,0.08)', tickfont: { color: '#8A837A', size: 11 } },
    yaxis: { gridcolor: 'rgba(31,28,24,0.06)', zerolinecolor: 'rgba(31,28,24,0.08)', tickfont: { color: '#8A837A', size: 11 } },
    legend: { bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#DDD8CE', borderwidth: 1, font: { color: '#5C5650', size: 11 } },
  };
}
function plotCfg() { return {displayModeBar:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d'],responsive:true}; }

/* ========================
   HELPERS
   ======================== */
function avg(arr)      { return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0; }
function fmtCur(v)     { return '$'+Number(v).toLocaleString('en-US',{maximumFractionDigits:0}); }
function fmtDate(d)    { return d instanceof Date ? d.toISOString().slice(0,10) : String(d); }
