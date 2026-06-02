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

document.addEventListener('DOMContentLoaded', () => {
  injectIllustrations();
  decorateIcons();
  decorateSidebar();
  decorateNavTabs();
  initAuthPage();
  initAppShellAnimation();
  initReveal();
  initUploadZone();
});

window.refreshReveals = refreshReveals;
window.animateStatRings = animateStatRings;
window.initAppShellAnimation = initAppShellAnimation;
window.initReveal = initReveal;
window.setSidebarUser = setSidebarUser;
window.decorateIcons = decorateIcons;
window.decorateSidebar = decorateSidebar;
