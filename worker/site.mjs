import {handleMcp} from './mcp.mjs';
const markdownType='text/markdown; charset=utf-8';
export function prefersMarkdown(accept=''){
 const ranges=accept.toLowerCase().split(',').map(part=>{const [type,...params]=part.trim().split(';');const q=params.find(p=>p.trim().startsWith('q='));const value=q?Number(q.trim().slice(2)):1;return {type,q:Number.isFinite(value)&&value>=0&&value<=1?value:0};});
 const quality=type=>{for(const match of [type,'text/*','*/*']){const values=ranges.filter(r=>r.type===match);if(values.length)return Math.max(...values.map(r=>r.q));}return 0;};
 return ranges.some(r=>r.type==='text/markdown'&&r.q>0)&&quality('text/markdown')>=quality('text/html');
}
export function apiError(status,code,message,hint){return Response.json({error:message,code,message,hint,docs:'https://zer.foo/agents/'},{status,headers:{'Cache-Control':'no-store'}});}
function finish(response,request,{alternate}={}){
 const headers=new Headers(response.headers);
 headers.set('X-Content-Type-Options','nosniff');headers.set('Referrer-Policy','strict-origin-when-cross-origin');
 const vary=new Set((headers.get('Vary')||'').split(',').map(s=>s.trim()).filter(Boolean));if(![...vary].some(s=>s.toLowerCase()==='accept'))vary.add('Accept');headers.set('Vary',[...vary].join(', '));
 headers.append('Link','</llms.txt>; rel="describedby"'+(alternate?`, <${alternate}>; rel="alternate"; type="text/markdown"`:''));
 if(new URL(request.url).pathname.startsWith('/create/'))headers.set('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; connect-src https://design.zer.foo; img-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
 return new Response(request.method==='HEAD'?null:response.body,{status:response.status,headers});
}
export default {async fetch(request,env){
 const url=new URL(request.url),path=url.pathname,md=prefersMarkdown(request.headers.get('Accept')||'');
 if(path==='/mcp'||path==='/mcp/')return finish(await handleMcp(request,env),request);
 if(['/create','/create/','/create/index.html'].includes(path))return finish(Response.redirect(new URL('/#message',url.origin),301),request);
 if(path==='/api'||path==='/api/'){
  if(!['GET','HEAD'].includes(request.method)){const error=apiError(405,'method_not_allowed','API discovery supports GET or HEAD.','Fetch /openapi.json to find supported operations.');error.headers.set('Allow','GET, HEAD');return finish(error,request);}
  return finish(Response.json({name:'Zerfoo ML framework',description:'Agent resources and the hosted model designer. Inference and training run on your own hardware.',openapi:'https://zer.foo/openapi.json',docs:'https://zer.foo/agents/',designer:'https://design.zer.foo/api/design',cli:'https://zer.foo/cli/',mcp:'https://zer.foo/mcp'},{headers:{'Cache-Control':'public, max-age=300'}}),request);
 }
 if(path.startsWith('/api/'))return finish(apiError(404,'not_found','No API operation exists at this URL.','Use /openapi.json. The hosted designer is at https://design.zer.foo/api/design.'),request);
 if(!['GET','HEAD'].includes(request.method)){const error=apiError(405,'method_not_allowed','This resource supports GET or HEAD.','Use the method documented in /openapi.json.');error.headers.set('Allow','GET, HEAD');return finish(error,request);}
 const directMarkdown=path.endsWith('.md')||['/llms.txt','/llms-full.txt','/agent-instructions.txt'].includes(path);
 let response=await env.ASSETS.fetch(request);
 if(md&&!directMarkdown&&response.status===304){
  const lookupHeaders=new Headers(request.headers);lookupHeaders.delete('If-None-Match');lookupHeaders.delete('If-Modified-Since');
  const source=await env.ASSETS.fetch(new Request(request,{headers:lookupHeaders}));
  if(source.headers.get('Content-Type')?.includes('text/html'))response=source;
  else return finish(response,request);
 }
 if(response.status===404){
  if(md)return finish(new Response('# 404 — Page not found\n\nThe requested Zerfoo resource does not exist. Check the URL or explore [documentation](https://zer.foo/docs/), [llms.txt](https://zer.foo/llms.txt), or the [sitemap](https://zer.foo/sitemap.xml).\n',{status:404,headers:{'Content-Type':markdownType,'Cache-Control':'no-store'}}),request);
  if((request.headers.get('Accept')||'').includes('application/json'))return finish(apiError(404,'not_found','The requested Zerfoo resource does not exist.','Check the URL or use https://zer.foo/llms.txt.'),request);
 }
 if(path.endsWith('.md')||path==='/llms.txt'||path==='/llms-full.txt'||path==='/agent-instructions.txt'){
  if(response.ok){const headers=new Headers(response.headers);headers.set('Content-Type',markdownType);return finish(new Response(response.body,{status:response.status,headers}),request);}
 }
 if(response.ok&&response.headers.get('Content-Type')?.includes('text/html')){
  const alternate=path.endsWith('.html')?path.replace(/[^/]+\.html$/,'index.md'):path.replace(/\/?$/,'/')+'index.md';
  if(md){const target=new URL(alternate,url);const selectedHeaders=new Headers({Accept:'text/markdown'});for(const key of ['If-None-Match','If-Modified-Since'])if(request.headers.has(key))selectedHeaders.set(key,request.headers.get(key));
   const result=await env.ASSETS.fetch(new Request(target,{headers:selectedHeaders}));
   if(result.status===304)return finish(result,request,{alternate});
   if(!result.ok)return finish(apiError(503,'markdown_unavailable','The Markdown representation is temporarily unavailable.','Retry with Accept: text/html or read /llms.txt.'),request);
   const headers=new Headers(result.headers);headers.set('Content-Type',markdownType);headers.delete('Content-Length');return finish(new Response(result.body,{status:200,headers}),request,{alternate});
  }
  return finish(response,request,{alternate});
 }
 return finish(response,request);
}};
