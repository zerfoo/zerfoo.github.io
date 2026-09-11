(() => {
  if (location.hostname !== 'zerfoo.feza.ai') return;
  const target = `https://zer.foo${location.pathname}${location.search}${location.hash}`;
  location.replace(target);
})();
