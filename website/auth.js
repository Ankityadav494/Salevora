/* Shared auth helpers for Salevora frontend */

const API_BASE = (() => {
  const origin = window.location.origin;
  if (origin.includes('localhost') || origin.includes('127.0.0.1')) {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  if (origin.startsWith('http')) return origin;
  return 'http://localhost:8000';
})();

const TOKEN_KEY   = 'salevora_token';
const SESSION_KEY = 'salevora_session';
let pendingOtp    = null;
let pendingReset  = null;

function getToken()  { return localStorage.getItem(TOKEN_KEY); }
function setToken(t) { localStorage.setItem(TOKEN_KEY, t); }

function getSession()   { return JSON.parse(localStorage.getItem(SESSION_KEY) || 'null'); }
function saveSession(s) { localStorage.setItem(SESSION_KEY, JSON.stringify(s)); }

function saveAlertFields(user) {
  saveSession({
    ...getSession(),
    name: user.name,
    email: user.email,
    alert_email: user.alert_email,
    alerts_enabled: user.alerts_enabled,
    alert_time: user.alert_time,
    alert_cooldown_hours: user.alert_cooldown_hours,
    alert_schedule_enabled: user.alert_schedule_enabled,
    email_verified: user.email_verified,
  });
}
function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(SESSION_KEY);
}

function authHeaders(extra = {}) {
  const headers = { ...extra };
  const token = getToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

async function authFetch(url, opts = {}) {
  const headers = authHeaders(opts.headers || {});
  const r = await fetch(url, { ...opts, headers });
  if (r.status === 401) {
    clearSession();
    if (typeof window.onAuthExpired === 'function') window.onAuthExpired();
  }
  return r;
}

async function validateSession() {
  if (!getToken()) return null;
  try {
    const r = await authFetch(`${API_BASE}/api/auth/me`);
    if (!r.ok) { clearSession(); return null; }
    const user = await r.json();
    saveAlertFields(user);
    return user;
  } catch {
    clearSession();
    return null;
  }
}

async function loginUser(email, password) {
  const r = await fetch(`${API_BASE}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof friendlyError === 'function' ? friendlyError(data.detail) : (data.detail || 'Login failed'));
  setToken(data.access_token);
  saveAlertFields(data.user);
  return data.user;
}

async function requestOtp(email, purpose, { password, name } = {}) {
  const body = { email, purpose };
  if (password) body.password = password;
  if (name) body.name = name;
  const r = await fetch(`${API_BASE}/api/auth/otp/request`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof friendlyError === 'function' ? friendlyError(data.detail) : (data.detail || 'Failed to send verification code'));
  pendingOtp = { email, purpose, password, name };
  return data;
}

async function verifyOtpAndLogin(email, otp, purpose) {
  const r = await fetch(`${API_BASE}/api/auth/otp/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, otp, purpose }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof friendlyError === 'function' ? friendlyError(data.detail) : (data.detail || 'Invalid verification code'));
  setToken(data.access_token);
  saveAlertFields(data.user);
  pendingOtp = null;
  return data.user;
}

function getPendingOtp() { return pendingOtp; }
function clearPendingOtp() { pendingOtp = null; }

async function registerUser(name, email, password) {
  const r = await fetch(`${API_BASE}/api/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof friendlyError === 'function' ? friendlyError(data.detail) : (data.detail || 'Registration failed'));
  setToken(data.access_token);
  saveAlertFields(data.user);
  return data.user;
}

async function requestPasswordReset(email) {
  const r = await fetch(`${API_BASE}/api/auth/forgot-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof friendlyError === 'function' ? friendlyError(data.detail) : (data.detail || 'Could not send reset code'));
  pendingReset = { email };
  return data;
}

async function resetPassword(email, otp, password) {
  const r = await fetch(`${API_BASE}/api/auth/reset-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, otp, password }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof friendlyError === 'function' ? friendlyError(data.detail) : (data.detail || 'Could not reset password'));
  setToken(data.access_token);
  saveAlertFields(data.user);
  pendingReset = null;
  return data.user;
}

function getPendingReset() { return pendingReset; }
function clearPendingReset() { pendingReset = null; }

async function updateAlertSettings(settings = {}) {
  const body = {};
  const keys = [
    'alert_email',
    'alerts_enabled',
    'alert_time',
    'alert_cooldown_hours',
    'alert_schedule_enabled',
  ];
  for (const key of keys) {
    if (settings[key] !== undefined) body[key] = settings[key];
  }
  const r = await authFetch(`${API_BASE}/api/auth/alert-settings`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || 'Failed to save alert settings');
  saveAlertFields(data);
  return data;
}

async function fetchAlertStatus() {
  try {
    const r = await authFetch(`${API_BASE}/api/alerts/status`);
    if (!r.ok) return { brevo_configured: false };
    return await r.json();
  } catch {
    return { brevo_configured: false };
  }
}

async function sendBrevoAlerts(force = false) {
  const r = await authFetch(`${API_BASE}/api/alerts/send?force=${force}`, { method: 'POST' });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof friendlyError === 'function' ? friendlyError(data.detail) : (data.detail || 'Failed to send reminders'));
  return data;
}

function notifyAlertEmail(alertResult) {
  const msg = typeof friendlyAlertMessage === 'function'
    ? friendlyAlertMessage(alertResult)
    : alertResult?.message;
  if (!msg) return;
  if (alertResult?.status === 'sent') toast(`📧 ${msg}`, 'success');
  else if (alertResult?.status === 'cooldown') toast(`📧 ${msg}`, 'warning');
}

window.onAuthExpired = function onAuthExpired() {
  const onApp = document.getElementById('appScreen')?.style.display !== 'none';
  const onAlerts = document.body?.classList.contains('page-alerts');
  if (onApp) {
    document.getElementById('appScreen').style.display = 'none';
    document.getElementById('authScreen').style.display = '';
    if (typeof toast === 'function') toast('Please sign in again.', 'warning');
  } else if (onAlerts) {
    window.location.href = 'index.html';
  }
};
