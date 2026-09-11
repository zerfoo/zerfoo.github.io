export default {
  fetch(request) {
    const source = new URL(request.url);
    const target = `https://zer.foo${source.pathname}${source.search}`;
    return new Response(null, {
      status: 301,
      headers: {
        Location: target,
        'Cache-Control': 'public, max-age=3600',
      },
    });
  },
};
