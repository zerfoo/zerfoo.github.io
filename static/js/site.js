(()=>{
 const root=document.documentElement,theme=document.querySelector('.theme-toggle');
 const system=matchMedia('(prefers-color-scheme: light)');
 const update=()=>{theme.setAttribute('aria-label',`Switch to ${root.dataset.theme==='dark'?'light':'dark'} mode`);document.querySelector('meta[name="theme-color"]').content=root.dataset.theme==='dark'?'#0c0c0c':'#fafaf8';};
 theme.onclick=()=>{root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';try{localStorage.setItem('zerfoo-theme',root.dataset.theme);}catch{}update();};update();
 system.addEventListener('change',()=>{let saved;try{saved=localStorage.getItem('zerfoo-theme');}catch{}if(!saved){root.dataset.theme=system.matches?'light':'dark';update();}});
 const menu=document.querySelector('#nav-menu'),toggle=document.querySelector('.menu-toggle');
 // Nonmodal dialog keeps the header's close button available; background is inert.
 const background=[document.querySelector('main'),document.querySelector('footer')];
 function setMenu(open,focus=false){toggle.setAttribute('aria-expanded',String(open));toggle.setAttribute('aria-label',open?'Close menu':'Open menu');document.body.classList.toggle('menu-open',open);background.forEach(el=>el.inert=open);if(open){menu.show();menu.querySelector('a').focus();}else{menu.close();if(focus)toggle.focus();}}
 toggle.onclick=()=>setMenu(!menu.open,true);menu.querySelectorAll('a').forEach(a=>a.onclick=()=>setMenu(false));
 document.addEventListener('keydown',e=>{if(!menu.open)return;if(e.key==='Escape'){e.preventDefault();setMenu(false,true);}if(e.key==='Tab'){const items=[theme,toggle,...menu.querySelectorAll('a')],i=items.indexOf(document.activeElement);if(e.shiftKey&&i<=0){e.preventDefault();items.at(-1).focus();}else if(!e.shiftKey&&i===items.length-1){e.preventDefault();items[0].focus();}}});
 matchMedia('(min-width:1025px)').addEventListener('change',e=>{if(e.matches&&menu.open)setMenu(false);});
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
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 async function entrance(){if(reduced.matches){root.classList.remove('entrance');return;}await Promise.race([document.fonts.ready,new Promise(r=>setTimeout(r,400))]);const anims=[],ease='cubic-bezier(.16,1,.3,1)',d=innerWidth<=680?.7:1;function lift(el,delay,duration=650,y=14){if(el)anims.push(el.animate([{opacity:0,transform:`translateY(${y*d}px)`},{opacity:1,transform:'translateY(0)'}],{duration,delay,easing:ease,fill:'both'}));}
 lift(document.querySelector('.wordmark'),0,600,0);document.querySelectorAll('.nav-links a').forEach((el,i)=>lift(el,100+i*50,500,10));lift(document.querySelector('.nav-actions'),260,500,10);
 document.querySelectorAll('.hl-line').forEach((el,i)=>anims.push(el.animate([{opacity:0,transform:'translateY(108%)'},{opacity:1,offset:.14},{opacity:1,transform:'translateY(0)'}],{duration:950,delay:200+i*90,easing:ease,fill:'both'})));
 lift(document.querySelector('.hero-description'),520);lift(document.querySelector('.composer-shell'),640,950,22);lift(document.querySelector('.examples'),1080,650,12);
 await Promise.allSettled(anims.map(a=>a.finished));root.classList.remove('entrance');anims.forEach(a=>a.cancel());root.classList.add('hero-ready');}
 entrance();
})();
