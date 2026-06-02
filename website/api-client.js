/* Salevora — unified API client (all backend routes) */

async function fetchJson(path, opts = {}) {
  const r = await fetch(`${API_BASE}${path}`, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const msg = typeof friendlyError === 'function'
      ? friendlyError(data.detail || data.message || r.statusText)
      : (data.detail || r.statusText);
    throw new Error(msg);
  }
  return data;
}

async function authFetchJson(path, opts = {}) {
  const r = await authFetch(`${API_BASE}${path}`, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const msg = typeof friendlyError === 'function'
      ? friendlyError(data.detail || data.message || r.statusText)
      : (data.detail || r.statusText);
    throw new Error(msg);
  }
  return data;
}

const SalevoraAPI = {
  health: () => fetchJson('/api/health'),
  appStatus: () => authFetch(`${API_BASE}/api/app/status`).then(r => r.json()),

  dataInfo: () => fetchJson('/data/info'),
  downloadData: () => fetchJson('/data/download'),

  forecast: (horizon, model) =>
    fetchJson(`/api/forecast?horizon=${horizon}&model=${encodeURIComponent(model)}`),
  analytics: () => fetchJson('/api/analytics/summary'),
  modelsStatus: () => fetchJson('/api/models/status'),

  uploadFile: async (file, mode = 'replace') => {
    const buf = await file.arrayBuffer();
    const r = await authFetch(
      `${API_BASE}/data/upload?filename=${encodeURIComponent(file.name)}&mode=${mode}`,
      { method: 'POST', body: buf, headers: { 'Content-Type': 'application/octet-stream' } }
    );
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(friendlyError?.(data.detail) || data.detail || 'Upload failed');
    return data;
  },

  inventory: {
    kpis: () => fetchJson('/api/inventory/kpis'),
    live: () => fetchJson('/api/inventory/live'),
    alerts: () => fetchJson('/api/inventory/alerts'),
    abc: () => fetchJson('/api/inventory/abc'),
    forecast: (sku) => fetchJson(`/api/inventory/forecast?sku=${encodeURIComponent(sku)}`),
    evaluate: () => authFetchJson('/api/inventory/evaluate', { method: 'POST' }),
    restock: (sku) => authFetchJson(`/api/inventory/restock/${encodeURIComponent(sku)}`, { method: 'POST' }),
  },

  alerts: {
    status: () => authFetchJson('/api/alerts/status'),
    evaluate: () => authFetchJson('/api/alerts/evaluate'),
    send: (force = false) => authFetchJson(`/api/alerts/send?force=${force}`, { method: 'POST' }),
    history: (limit = 20) => authFetchJson(`/api/alerts/history?limit=${limit}`),
  },
};

async function requireAuth(redirect = true) {
  const user = await validateSession();
  if (!user && redirect) {
    window.location.href = 'index.html';
    return null;
  }
  return user;
}
