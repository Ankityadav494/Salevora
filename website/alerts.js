/* Salevora — Email alerts settings page */

let alertSchedulerId = null;
let currentUser = null;

function toast(msg, type = 'success', ms = 3500) {
  const c = document.getElementById('toastContainer');
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transition = 'opacity 0.3s';
    setTimeout(() => t.remove(), 350);
  }, ms);
}

function fmtTime12(h, m) {
  const ampm = h >= 12 ? 'PM' : 'AM';
  const h12 = h % 12 || 12;
  return `${h12}:${String(m).padStart(2, '0')} ${ampm}`;
}

function schedulerStorageKey(email) {
  return `salevora_alert_last_run_${email || 'anon'}`;
}

function readFormSettings() {
  return {
    alert_email: document.getElementById('alertEmailTo')?.value.trim(),
    alerts_enabled: document.getElementById('alertEmailEnabled')?.value === '1',
    alert_time: document.getElementById('alertTime')?.value || '10:00',
    alert_schedule_enabled: document.getElementById('alertScheduleEnabled')?.value === '1',
    alert_cooldown_hours: parseInt(document.getElementById('alertCooldown')?.value, 10) || 24,
  };
}

function fillForm(user) {
  const emailEl = document.getElementById('alertEmailTo');
  const enabledEl = document.getElementById('alertEmailEnabled');
  const timeEl = document.getElementById('alertTime');
  const scheduleEl = document.getElementById('alertScheduleEnabled');
  const cooldownEl = document.getElementById('alertCooldown');

  if (emailEl) emailEl.value = user.alert_email || user.email || '';
  if (enabledEl) enabledEl.value = user.alerts_enabled === false ? '0' : '1';
  if (timeEl) timeEl.value = user.alert_time || '10:00';
  if (scheduleEl) scheduleEl.value = user.alert_schedule_enabled === false ? '0' : '1';
  if (cooldownEl) cooldownEl.value = user.alert_cooldown_hours ?? 24;
}

function updateScheduleStatus(user) {
  const el = document.getElementById('scheduleStatus');
  if (!el) return;
  if (!user.alerts_enabled) {
    el.textContent = 'Email reminders are turned off.';
    return;
  }
  if (!user.alert_schedule_enabled) {
    el.textContent = 'Daily schedule is off — use “Send now” when you want a reminder.';
    return;
  }
  const [h, m] = (user.alert_time || '10:00').split(':').map(Number);
  el.textContent = `Daily send scheduled for ${fmtTime12(h, m)} (local time). Keep this tab open for automatic sends.`;
}

async function loadBrevoStatus() {
  try {
    const status = await SalevoraAPI.alerts.status();
    if (status.settings) {
      fillForm({ ...currentUser, ...status.settings });
      updateScheduleStatus({ ...currentUser, ...status.settings });
    }
  } catch {
    /* settings form still uses saved user prefs */
  }
}

async function loadAlertHistory() {
  const list = document.getElementById('alertHistoryList');
  if (!list || !getToken()) return;
  try {
    const { history } = await SalevoraAPI.alerts.history(30);
    const mine = (history || []).filter(h => {
      const to = (h.to || h.to_email || '').toLowerCase();
      const alertEmail = (currentUser?.alert_email || currentUser?.email || '').toLowerCase();
      return !h.user || h.user === currentUser?.email || to === alertEmail;
    });
    if (!mine.length) {
      list.innerHTML = '<div class="alert-history-empty">No emails sent yet.</div>';
      return;
    }
    list.innerHTML = mine.map(h => {
      const when = h.sent_at ? new Date(h.sent_at).toLocaleString() : '—';
      const count = h.alert_count ?? h.count ?? '?';
      const to = h.to || h.to_email || 'you';
      return `<div class="alert-history-item"><span class="alert-history-when">${when}</span><span class="alert-history-detail">${count} item(s) → ${to}</span></div>`;
    }).join('');
  } catch {
    list.innerHTML = '<div class="alert-history-empty">Could not load history.</div>';
  }
}

async function saveAlertSettings(e) {
  e.preventDefault();
  const resultEl = document.getElementById('alertSaveResult');
  const btn = document.getElementById('saveAlertsBtn');
  const settings = readFormSettings();
  btn.disabled = true;
  resultEl.style.display = '';
  resultEl.textContent = 'Saving…';
  resultEl.className = 'alert-result';
  try {
    const user = await updateAlertSettings(settings);
    currentUser = user;
    updateScheduleStatus(user);
    startDailyScheduler(user);
    resultEl.className = 'alert-result ok';
    resultEl.textContent = 'Settings saved.';
    toast('Alert settings saved.', 'success');
  } catch (err) {
    resultEl.className = 'alert-result warn';
    resultEl.textContent = friendlyError(err.message);
    toast(friendlyError(err.message), 'warning');
  } finally {
    btn.disabled = false;
  }
}

async function sendAlertsNow(force = false) {
  const resultEl = document.getElementById('alertSaveResult');
  resultEl.style.display = '';
  resultEl.textContent = 'Sending…';
  resultEl.className = 'alert-result';
  try {
    const settings = readFormSettings();
    if (settings.alert_email !== currentUser?.alert_email || settings.alerts_enabled !== currentUser?.alerts_enabled) {
      currentUser = await updateAlertSettings(settings);
    }
    const res = await sendBrevoAlerts(force);
    resultEl.className = 'alert-result ' + (res.status === 'sent' ? 'ok' : 'warn');
    resultEl.textContent = friendlyAlertMessage(res) || 'Done.';
    toast(friendlyAlertMessage(res) || 'Done.', res.status === 'sent' ? 'success' : 'warning');
    if (res.status === 'sent') loadAlertHistory();
  } catch (err) {
    resultEl.className = 'alert-result warn';
    resultEl.textContent = friendlyError(err.message);
    toast(friendlyError(err.message), 'warning');
  }
}

function startDailyScheduler(user) {
  if (alertSchedulerId) {
    clearInterval(alertSchedulerId);
    alertSchedulerId = null;
  }
  if (!user?.alerts_enabled || !user?.alert_schedule_enabled) return;

  const [h, m] = (user.alert_time || '10:00').split(':').map(Number);
  const key = schedulerStorageKey(user.email);

  alertSchedulerId = setInterval(async () => {
    const now = new Date();
    if (now.getHours() !== h || now.getMinutes() !== m) return;

    const today = now.toISOString().slice(0, 10);
    if (localStorage.getItem(key) === today) return;

    try {
      const res = await sendBrevoAlerts(false);
      localStorage.setItem(key, today);
      notifyAlertEmail(res);
      loadAlertHistory();
    } catch {
      /* silent — user can send manually */
    }
  }, 60000);
}

function handleAlertsLogout() {
  if (alertSchedulerId) clearInterval(alertSchedulerId);
  clearSession();
  window.location.href = 'index.html';
}

window.addEventListener('DOMContentLoaded', async () => {
  currentUser = await requireAuth(true);
  if (!currentUser) return;

  if (typeof decorateIcons === 'function') decorateIcons();
  if (typeof decorateSidebar === 'function') decorateSidebar();
  if (typeof setSidebarUser === 'function') setSidebarUser(currentUser);

  fillForm(currentUser);
  updateScheduleStatus(currentUser);
  await loadBrevoStatus();
  await loadAlertHistory();
  startDailyScheduler(currentUser);
});
