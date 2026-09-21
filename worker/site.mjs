export default {async fetch(request,env){
 const url=new URL(request.url);
 if(['/create','/create/','/create/index.html'].includes(url.pathname))return Response.redirect(new URL('/#message',url.origin),301);
 const response=await env.ASSETS.fetch(request);const headers=new Headers(response.headers);
 headers.set('X-Content-Type-Options','nosniff');headers.set('Referrer-Policy','strict-origin-when-cross-origin');
 if(new URL(request.url).pathname.startsWith('/create/'))headers.set('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; connect-src https://design.zer.foo; img-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
 return new Response(response.body,{status:response.status,headers});
}};
