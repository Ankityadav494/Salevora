/* Salevora — inline SVG icons (Lucide-style, stroke) */

const ICON_PATHS = {
  chart: '<path d="M3 3v18h18"/><path d="M7 16l4-8 4 5 5-9"/>',
  trending: '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
  mail: '<rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
  package: '<path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
  bell: '<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
  logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>',
  dashboard: '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
  boxes: '<path d="M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z"/><path d="m7 16.5-4.74-2.85"/><path d="m7 16.5 5-3"/><path d="M7 16.5v5.17"/><path d="M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z"/><path d="m17 16.5-5-3"/><path d="m17 16.5 4.74-2.85"/><path d="M17 16.5v5.17"/><path d="M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z"/><path d="M12 8 7.26 5.15"/><path d="m12 8 4.74-2.85"/><path d="M12 13.5V8"/>',
  calendar: '<rect width="18" height="18" x="3" y="4" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
  target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
  dollar: '<line x1="12" y1="2" x2="12" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
  cart: '<circle cx="8" cy="21" r="1"/><circle cx="19" cy="21" r="1"/><path d="M2.05 2.05h2l2.66 12.42a2 2 0 0 0 2 1.58h9.78a2 2 0 0 0 1.95-1.57l1.65-7.43H5.12"/>',
  layers: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
  sparkles: '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/>',
  alert: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  refresh: '<path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/>',
  send: '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
  user: '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
  lock: '<rect width="18" height="11" x="3" y="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
  tag: '<path d="M12 2H2v10l9.29 9.29a1 1 0 0 0 1.41 0l6.59-6.59a1 1 0 0 0 0-1.41L12 2Z"/><circle cx="7" cy="7" r="1.5"/>',
  clock: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
  warehouse: '<path d="M22 8.35V20a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V8.35A2 2 0 0 1 3.26 6.5l8-3.2a2 2 0 0 1 1.48 0l8 3.2A2 2 0 0 1 22 8.35Z"/><path d="M6 18h12"/><path d="M6 14h12"/><path d="M12 12v6"/>',
};

function svIcon(name, sizeOrOpts = 20, className = '') {
  let size = 20;
  let cls = className;
  if (typeof sizeOrOpts === 'object') {
    size = sizeOrOpts.size ?? 20;
    cls = sizeOrOpts.className ?? '';
  } else if (typeof sizeOrOpts === 'number') {
    size = sizeOrOpts;
  }
  const paths = ICON_PATHS[name];
  if (!paths) return '';
  return `<svg class="sv-icon ${cls}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;
}

const KPI_ICON_MAP = {
  'Total revenue': 'dollar',
  'Total sales': 'cart',
  'Daily average': 'calendar',
  'Last 4 weeks': 'trending',
  'Month over month': 'chart',
  'Year over year': 'layers',
  'Top category': 'tag',
  'Period start': 'calendar',
  'Period end': 'clock',
  'Categories': 'layers',
};

const INV_KPI_ICONS = ['package', 'dollar', 'alert', 'bell', 'clock', 'check'];

function kpiIconFor(label) {
  return KPI_ICON_MAP[label] || 'chart';
}

function authIllustration() {
  return `<svg class="auth-scene" viewBox="0 0 400 320" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <circle class="auth-orb auth-orb-1" cx="320" cy="60" r="80" fill="rgba(255,255,255,0.06)"/>
    <circle class="auth-orb auth-orb-2" cx="60" cy="260" r="55" fill="rgba(255,255,255,0.05)"/>
    <g class="auth-float auth-float-1">
      <rect x="48" y="72" width="168" height="112" rx="14" fill="rgba(255,255,255,0.14)" stroke="rgba(255,255,255,0.25)"/>
      <rect x="68" y="96" width="48" height="8" rx="4" fill="rgba(255,255,255,0.5)"/>
      <rect x="68" y="114" width="80" height="6" rx="3" fill="rgba(255,255,255,0.25)"/>
      <rect x="68" y="148" width="20" height="24" rx="4" fill="rgba(255,255,255,0.55)" class="auth-bar auth-bar-1"/>
      <rect x="96" y="132" width="20" height="40" rx="4" fill="rgba(255,255,255,0.75)" class="auth-bar auth-bar-2"/>
      <rect x="124" y="140" width="20" height="32" rx="4" fill="rgba(255,255,255,0.45)" class="auth-bar auth-bar-3"/>
      <rect x="152" y="124" width="20" height="48" rx="4" fill="rgba(255,255,255,0.85)" class="auth-bar auth-bar-4"/>
      <path d="M68 178 L188 178" stroke="rgba(255,255,255,0.2)" stroke-width="2"/>
    </g>
    <g class="auth-float auth-float-2">
      <rect x="228" y="168" width="120" height="96" rx="12" fill="rgba(255,255,255,0.12)" stroke="rgba(255,255,255,0.22)"/>
      <path d="M252 208 L276 192 L300 216 L324 188" stroke="rgba(255,255,255,0.7)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="324" cy="188" r="5" fill="#fff"/>
      <rect x="252" y="228" width="72" height="6" rx="3" fill="rgba(255,255,255,0.3)"/>
    </g>
    <g class="auth-float auth-float-3">
      <rect x="248" y="48" width="88" height="72" rx="10" fill="rgba(255,255,255,0.1)" stroke="rgba(255,255,255,0.2)"/>
      <path d="M268 88 L284 72 L300 96 L316 80" stroke="rgba(255,255,255,0.55)" stroke-width="2.5" stroke-linecap="round"/>
      <rect x="268" y="100" width="48" height="5" rx="2.5" fill="rgba(255,255,255,0.25)"/>
    </g>
    <g class="auth-float auth-float-4">
      <circle cx="88" cy="228" r="28" fill="rgba(255,255,255,0.15)" stroke="rgba(255,255,255,0.3)"/>
      <path d="M78 228 L86 236 L98 220" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    </g>
  </svg>`;
}

function uploadIllustration() {
  return `<svg class="hero-scene" viewBox="0 0 280 160" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <rect x="20" y="40" width="100" height="100" rx="12" fill="var(--accent-soft)" stroke="var(--border)" class="hero-float hero-float-1"/>
    <path d="M44 88 L60 72 L76 88 L92 68" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <rect x="160" y="24" width="96" height="116" rx="10" fill="var(--surface)" stroke="var(--border)" class="hero-float hero-float-2"/>
    <rect x="176" y="44" width="64" height="8" rx="4" fill="var(--border)"/>
    <rect x="176" y="60" width="48" height="6" rx="3" fill="var(--surface-2)"/>
    <rect x="176" y="88" width="16" height="28" rx="3" fill="var(--accent-muted)" opacity="0.6"/>
    <rect x="196" y="76" width="16" height="40" rx="3" fill="var(--accent)" opacity="0.85"/>
    <rect x="216" y="84" width="16" height="32" rx="3" fill="var(--accent-muted)" opacity="0.45"/>
    <circle cx="140" cy="80" r="22" fill="var(--accent)" class="hero-float hero-float-3"/>
    <path d="M132 80 L138 86 L150 72" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;
}

function stockIllustration() {
  return `<svg class="hero-scene hero-scene--stock" viewBox="0 0 240 140" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <rect x="8" y="48" width="72" height="72" rx="8" fill="var(--accent-soft)" stroke="var(--border)" class="hero-float hero-float-1"/>
    <rect x="24" y="64" width="40" height="32" rx="4" fill="var(--accent)" opacity="0.25"/>
    <path d="M8 48 L44 24 L80 48" stroke="var(--accent-muted)" stroke-width="2" fill="none"/>
    <rect x="96" y="56" width="64" height="64" rx="8" fill="var(--surface)" stroke="var(--border)" class="hero-float hero-float-2"/>
    <rect x="112" y="72" width="32" height="32" rx="4" fill="var(--accent-soft)"/>
    <rect x="176" y="40" width="56" height="80" rx="8" fill="var(--surface)" stroke="var(--border)" class="hero-float hero-float-3"/>
    <rect x="192" y="56" width="24" height="48" rx="3" fill="var(--yellow-soft)" stroke="#FDE68A"/>
    <circle cx="204" cy="72" r="6" fill="var(--yellow)"/>
  </svg>`;
}

window.svIcon = svIcon;
window.kpiIconFor = kpiIconFor;
window.INV_KPI_ICONS = INV_KPI_ICONS;
window.authIllustration = authIllustration;
window.uploadIllustration = uploadIllustration;
window.stockIllustration = stockIllustration;
