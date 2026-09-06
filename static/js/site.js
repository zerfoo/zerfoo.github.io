(() => {
  const menu = document.querySelector('#nav-menu');
  const toggle = document.querySelector('.menu-toggle');
  const desktop = matchMedia('(min-width: 769px)');
  let previousOverflow = '';
  function setMenu(open, restoreFocus = false) {
    const wasOpen = toggle.getAttribute('aria-expanded') === 'true';
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    menu.classList.toggle('active', open);
    if (open && !wasOpen) { previousOverflow = document.body.style.overflow; document.body.style.overflow = 'hidden'; }
    if (!open && wasOpen) document.body.style.overflow = previousOverflow;
    if (!open && restoreFocus) toggle.focus();
  }
  toggle.addEventListener('click', () => setMenu(toggle.getAttribute('aria-expanded') !== 'true'));
  menu.querySelectorAll('a').forEach(a => a.addEventListener('click', () => setMenu(false)));
  desktop.addEventListener('change', () => { if (desktop.matches) setMenu(false); });
  document.addEventListener('keydown', e => {
    if (toggle.getAttribute('aria-expanded') !== 'true') return;
    if (e.key === 'Escape') setMenu(false, true);
    if (e.key === 'Tab') {
      const links = [toggle, ...menu.querySelectorAll('a')];
      const current = links.indexOf(document.activeElement);
      if (e.shiftKey && current <= 0) { e.preventDefault(); links.at(-1).focus(); }
      else if (!e.shiftKey && (current === links.length - 1 || current === -1)) { e.preventDefault(); toggle.focus(); }
    }
  });
  let toastTimer;
  async function copy(text) {
    const toast = document.querySelector('.toast');
    try { await navigator.clipboard.writeText(text); toast.textContent = 'Copied to clipboard'; }
    catch { toast.textContent = 'Copy unavailable. Select the command and copy it manually.'; }
    toast.classList.add('visible'); clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('visible'), 3500);
  }
  document.querySelectorAll('[data-copy]').forEach(b => b.addEventListener('click', () => copy(b.dataset.copy)));
  const tabs = [...document.querySelectorAll('[role=tab]')];
  function selectTab(tab, focus = false) {
    tabs.forEach(t => { const active = t === tab; t.setAttribute('aria-selected', String(active)); t.tabIndex = active ? 0 : -1; document.getElementById(t.getAttribute('aria-controls')).hidden = !active; });
    if (focus) tab.focus();
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => selectTab(tab));
    tab.addEventListener('keydown', e => {
      const index = e.key === 'ArrowRight' ? (i + 1) % tabs.length : e.key === 'ArrowLeft' ? (i + tabs.length - 1) % tabs.length : e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : -1;
      if (index >= 0) { e.preventDefault(); selectTab(tabs[index], true); }
    });
  });
  document.querySelector('.code-copy').addEventListener('click', () => copy(document.querySelector('[role=tabpanel]:not([hidden]) code').textContent));
  const pipeline = document.querySelector('.pipeline');
  const nodes = ['node-model', 'node-runtime', 'node-app'].map(id => document.getElementById(id));
  const paths = [...document.querySelectorAll('.beam')];
  const gradient = document.getElementById('beam-gradient');
  const splash = document.querySelector('.splash');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let points = [], frame = 0, state = 'p1', stateStart = 0, visible = true;
  function measure() {
    const box = pipeline.getBoundingClientRect();
    points = nodes.map(n => { const r = n.getBoundingClientRect(); return {x:r.left+r.width/2-box.left,y:r.top+r.height/2-box.top}; });
    const d = points.map((p,i) => `${i ? 'L' : 'M'} ${p.x},${p.y}`).join(' ');
    paths.forEach(p => p.setAttribute('d',d));
  }
  function clean() { nodes.forEach(n => n.classList.remove('active')); splash.classList.remove('animate'); paths.forEach(p => p.style.opacity = '0'); }
  function tick(now) {
    if (!stateStart) stateStart = now;
    const elapsed = now - stateStart, progress = Math.min(elapsed / 800, 1);
    let percentage = state === 'p1' ? progress / 2 : .5 + progress / 2;
    if (state === 'p1' || state === 'p2') {
      const x = points[0].x + percentage * (points[2].x-points[0].x);
      gradient.setAttribute('x1',String(x-24)); gradient.setAttribute('x2',String(x+24));
      gradient.setAttribute('y1','0'); gradient.setAttribute('y2','0');
      paths.forEach((p,i) => p.style.opacity = i ? '1' : '.6');
      nodes[0].classList.toggle('active',state === 'p1' && progress < .4);
      nodes[2].classList.toggle('active',state === 'p2' && progress > .6);
      if (elapsed >= 800) {
        clean(); stateStart = now;
        if (state === 'p1') { state = 'splash'; splash.classList.add('animate'); }
        else state = 'idle';
      }
    } else if (state === 'splash' && elapsed >= 800) { clean(); state = 'p2'; stateStart = now; }
    else if (state === 'idle' && elapsed >= 1000) { state = 'p1'; stateStart = now; }
    frame = requestAnimationFrame(tick);
  }
  function sync() { cancelAnimationFrame(frame); clean(); state = 'p1'; stateStart = 0; if (!motion.matches && !document.hidden && visible) { measure(); frame = requestAnimationFrame(tick); } }
  const observer = new IntersectionObserver(entries => { visible = entries[0].isIntersecting; sync(); });
  observer.observe(pipeline);
  window.addEventListener('resize', measure);
  motion.addEventListener('change', sync);
  document.addEventListener('visibilitychange', sync);
  window.addEventListener('pagehide', () => { cancelAnimationFrame(frame); });
  window.addEventListener('pageshow', sync);
  measure(); sync();
})();
