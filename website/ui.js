/* Salevora — scroll reveals, page entrances & micro-interactions */

const motionOk = () => !window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function initReveal() {
  if (!motionOk()) {
    document.querySelectorAll('.reveal, .reveal-stagger > *').forEach(el => el.classList.add('is-visible'));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        if (entry.target.classList.contains('stat-ring-card')) {
          animateRingCard(entry.target);
        }
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -32px 0px' }
  );

  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
  document.querySelectorAll('.reveal-stagger').forEach(group => {
    [...group.children].forEach((child, i) => {
      child.classList.add('reveal');
      child.style.setProperty('--reveal-delay', `${i * 0.08}s`);
      observer.observe(child);
    });
  });
}

function refreshReveals() {
  if (!motionOk()) {
    document.querySelectorAll('#resultsWrap .reveal, #resultsWrap .stat-ring-card').forEach(el => {
      el.classList.add('is-visible', 'is-animated');
    });
    return;
  }

  document.querySelectorAll('#resultsWrap .app-section, #resultsWrap .chart-card, #resultsWrap .kpi-card').forEach((el, i) => {
    if (!el.classList.contains('reveal')) {
      el.classList.add('reveal');
      el.style.setProperty('--reveal-delay', `${Math.min(i * 0.05, 0.4)}s`);
    }
    requestAnimationFrame(() => {
      const obs = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            obs.disconnect();
          }
        },
        { threshold: 0.08 }
      );
      obs.observe(el);
    });
  });

  animateStatRings();
}

function initAuthPage() {
  const formWrap = document.querySelector('.auth-form-wrap.reveal');
  if (formWrap) {
    requestAnimationFrame(() => formWrap.classList.add('is-visible'));
  }
  if (!motionOk()) return;
  document.querySelectorAll('.auth-stagger > *').forEach((el, i) => {
    el.style.setProperty('--auth-delay', `${0.1 + i * 0.08}s`);
  });
}

function initAppShellAnimation(root) {
  const shell = root || document.querySelector('.app-shell');
  if (!shell) return;

  shell.querySelectorAll('.sidebar-link').forEach((link, i) => {
    link.style.setProperty('--nav-i', i);
  });

  if (!motionOk()) {
    shell.classList.add('is-ready');
    return;
  }

  requestAnimationFrame(() => {
    shell.classList.add('is-ready');
  });
}

function animateRingCard(card) {
  if (!motionOk() || card.dataset.ringDone) return;
  card.dataset.ringDone = '1';
  card.classList.add('is-animated');

  const prog = card.querySelector('.ring-chart circle:nth-child(2)');
  if (!prog) return;

  const parts = (prog.getAttribute('stroke-dasharray') || '0 264').split(/[\s,]+/);
  const target = parseFloat(parts[0]) || 0;
  const circ = parseFloat(parts[1]) || 264;

  prog.setAttribute('stroke-dasharray', `0 ${circ}`);
  requestAnimationFrame(() => {
    prog.style.transition = 'stroke-dasharray 1.1s cubic-bezier(0.22, 1, 0.36, 1)';
    prog.setAttribute('stroke-dasharray', `${target} ${circ}`);
  });
}

function animateStatRings() {
  const row = document.getElementById('ringStats');
  if (!row) return;

  row.querySelectorAll('.stat-ring-card').forEach((card, i) => {
    card.style.setProperty('--card-delay', `${i * 0.1}s`);
    card.classList.add('reveal');
    card.style.setProperty('--reveal-delay', `${i * 0.1}s`);

    if (!motionOk()) {
      card.classList.add('is-visible', 'is-animated');
      return;
    }

    requestAnimationFrame(() => {
      const obs = new IntersectionObserver(
        ([entry]) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add('is-visible');
          animateRingCard(entry.target);
          obs.disconnect();
        },
        { threshold: 0.2 }
      );
      obs.observe(card);
    });
  });
}

function setSidebarUser(user) {
  if (!user?.name) return;
  const sidebarName = document.getElementById('sidebarUserName');
  if (sidebarName) sidebarName.textContent = user.name;
  const initials = user.name.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase();
  const avatar = document.getElementById('sidebarAvatar');
  if (avatar) {
    avatar.innerHTML = `<span aria-hidden="true">${initials}</span>`;
    avatar.classList.add('avatar-pop');
    setTimeout(() => avatar.classList.remove('avatar-pop'), 600);
  }
}

function initUploadZone() {
  const zone = document.getElementById('dropZone');
  if (!zone) return;
  zone.addEventListener('dragenter', () => zone.classList.add('is-dragging'));
  zone.addEventListener('dragleave', (e) => {
    if (!zone.contains(e.relatedTarget)) zone.classList.remove('is-dragging');
  });
  zone.addEventListener('drop', () => zone.classList.remove('is-dragging'));
}

function decorateIcons() {
  if (typeof svIcon !== 'function') return;

  const authFeatureIcons = ['chart', 'trending', 'mail', 'file'];
  document.querySelectorAll('.af-item').forEach((item, i) => {
    const slot = item.querySelector('.af-icon');
    if (slot) slot.innerHTML = svIcon(authFeatureIcons[i] || 'check', 18);
  });

  document.querySelectorAll('[data-icon]').forEach(el => {
    if (el.classList.contains('sidebar-link') || el.classList.contains('sidebar-link-icon')) return;
    const name = el.getAttribute('data-icon');
    const size = parseInt(el.getAttribute('data-size') || '18', 10);
    el.innerHTML = svIcon(name, size);
  });

  document.querySelectorAll('.auth-tab-icon').forEach(el => {
    el.innerHTML = svIcon(el.getAttribute('data-icon'), 16);
  });
}

function injectIllustrations() {
  // Logo images are embedded in HTML; no SVG placeholders needed.
}

function decorateSidebar() {
  if (typeof svIcon !== 'function') return;
  document.querySelectorAll('.sidebar-link-icon[data-icon]').forEach(iconEl => {
    iconEl.innerHTML = svIcon(iconEl.getAttribute('data-icon'), 22);
  });
}

function decorateNavTabs() {
  if (typeof svIcon !== 'function') return;
  document.querySelectorAll('.nav-tab').forEach(tab => {
    if (tab.querySelector('.nav-tab-icon')) return;
    const href = tab.getAttribute('href') || '';
    const icon = href.includes('inventory') ? 'boxes' : href.includes('alerts') ? 'mail' : 'dashboard';
    const label = tab.textContent.trim();
    tab.innerHTML = `<span class="nav-tab-icon">${svIcon(icon, 16)}</span>${label}`;
  });
}

function resizeCharts() {
  if (typeof Plotly === 'undefined' || !Plotly.Plots?.resize) return;
  document.querySelectorAll('.chart-area, .chart-area-sm').forEach(el => {
    if (el.id && el.querySelector('.plotly')) Plotly.Plots.resize(el);
  });
}

function mobileNavQuery() {
  return window.matchMedia('(max-width: 1024px)');
}

function mobilePageTitle() {
  if (document.body.classList.contains('page-alerts')) return 'Email alerts';
  if (document.body.classList.contains('page-stock')) return 'Stock levels';
  if (document.body.classList.contains('page-dashboard')) return 'Dashboard';
  return 'Salevora';
}

function syncMobileLayout() {
  const mobile = mobileNavQuery().matches;
  document.body.classList.toggle('mobile-layout', mobile);
  return mobile;
}

function initMobileNav() {
  const sidebar = document.querySelector('.sidebar');
  const header = document.querySelector('.dash-header');
  if (!sidebar || !header) return;

  if (!sidebar.dataset.mobileNavReady) {
    sidebar.dataset.mobileNavReady = '1';
    bindMobileNav(sidebar, header);
  }

  syncMobileLayout();
  closeMobileNav();
}

function bindMobileNav(sidebar, header) {
  const backdrop = document.querySelector('.sidebar-backdrop');
  const toggle = header.querySelector('.sidebar-toggle');
  const closeBtn = sidebar.querySelector('.sidebar-close');
  if (!backdrop || !toggle || !closeBtn) return;

  function setToggleIcon(open) {
    if (typeof svIcon !== 'function') {
      toggle.textContent = open ? '×' : '☰';
      return;
    }
    toggle.innerHTML = svIcon(open ? 'x' : 'menu', 20);
  }

  function closeNav() {
    sidebar.classList.remove('is-open');
    backdrop.classList.remove('is-visible');
    toggle.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('aria-label', 'Open menu');
    document.body.classList.remove('nav-open');
    setToggleIcon(false);
    resizeCharts();
  }

  function openNav() {
    if (!syncMobileLayout()) return;
    sidebar.classList.add('is-open');
    backdrop.classList.add('is-visible');
    toggle.setAttribute('aria-expanded', 'true');
    toggle.setAttribute('aria-label', 'Close menu');
    document.body.classList.add('nav-open');
    setToggleIcon(true);
  }

  window.closeMobileNav = closeNav;
  window.openMobileNav = openNav;

  setToggleIcon(false);
  toggle.addEventListener('click', () => {
    if (sidebar.classList.contains('is-open')) closeNav();
    else openNav();
  });
  closeBtn.addEventListener('click', closeNav);
  backdrop.addEventListener('click', closeNav);
  sidebar.querySelectorAll('.sidebar-link').forEach(link => {
    link.addEventListener('click', () => {
      if (mobileNavQuery().matches) closeNav();
    });
  });

  window.addEventListener('resize', () => {
    syncMobileLayout();
    if (!mobileNavQuery().matches) closeNav();
    resizeCharts();
  });
}

function closeMobileNav() {
  if (typeof window.closeMobileNav === 'function') window.closeMobileNav();
  else {
    document.querySelector('.sidebar')?.classList.remove('is-open');
    document.querySelector('.sidebar-backdrop')?.classList.remove('is-visible');
    document.body.classList.remove('nav-open');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  injectIllustrations();
  decorateIcons();
  decorateSidebar();
  decorateNavTabs();
  initAuthPage();
  initAppShellAnimation();
  initMobileNav();
  initReveal();
  initUploadZone();
  syncMobileLayout();
});

window.refreshReveals = refreshReveals;
window.resizeCharts = resizeCharts;
window.initMobileNav = initMobileNav;
window.closeMobileNav = closeMobileNav;
window.syncMobileLayout = syncMobileLayout;
window.animateStatRings = animateStatRings;
window.initAppShellAnimation = initAppShellAnimation;
window.initReveal = initReveal;
window.setSidebarUser = setSidebarUser;
window.decorateIcons = decorateIcons;
window.decorateSidebar = decorateSidebar;
