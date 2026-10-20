import test from 'node:test';import assert from 'node:assert/strict';import worker,{prefersMarkdown} from '../worker/site.mjs';
const assets=new Map([['/','<!doctype html><html><main>Designer</main></html>'],['/docs/','<!doctype html><html>Docs</html>'],['/index.md','# Zerfoo\n\nDesign models with the Zerfoo ML framework.'],['/docs/index.md','# Zerfoo documentation\n\nRead the API reference.'],['/llms.txt','# Zerfoo\n\nAgent instructions.']]);
const env={ASSETS:{fetch:async request=>{const path=new URL(request.url).pathname;return new Response(assets.get(path)||'<h1>Not found</h1>',{status:assets.has(path)?200:404,headers:{'Content-Type':path.endsWith('.md')?'application/octet-stream':'text/html'}});}}};
const fetch=(path,accept='text/html',method='GET')=>worker.fetch(new Request('https://zer.foo'+path,{method,headers:{Accept:accept}}),env);
test('Accept quality and exclusions preserve browser HTML while explicitly negotiating Markdown',()=>{
 for(const value of ['text/markdown','text/markdown; q=1, text/html; q=.5','text/markdown, */*;q=.8'])assert.equal(prefersMarkdown(value),true);
 for(const value of ['', '*/*','text/html','text/markdown;q=0','text/markdown;q=.1,text/html;q=.9','text/markdown;q=0,*/*;q=1'])assert.equal(prefersMarkdown(value),false);
});
test('homepage and docs negotiate actual Markdown and advertise alternatives; HTML stays HTML',async()=>{
 for(const path of ['/','/docs/'])for(const accept of ['text/markdown','text/html']){const r=await fetch(path,accept);assert.equal(r.status,200);assert.match(r.headers.get('content-type'),new RegExp(accept));assert.match(r.headers.get('vary'),/Accept/);assert.match(r.headers.get('link'),/rel="alternate"/);assert.match(await r.text(),accept==='text/markdown'?/^# Zerfoo/:/<!doctype html>/);}
});
test('Markdown 404 has correct final status, type, explanation and discovery links',async()=>{
 const r=await fetch('/__missing','text/markdown');assert.equal(r.status,404);assert.match(r.headers.get('content-type'),/text\/markdown/);assert.match(r.headers.get('vary'),/Accept/);const body=await r.text();assert.ok(body.length>20);assert.match(body,/\[llms.txt\]\(https:\/\/zer.foo\/llms.txt\)/);assert.doesNotMatch(body,/<html/);
 assert.equal((await fetch('/__missing')).status,404);
});
test('API discovery is real JSON and unknown operations return structured resolution hints',async()=>{
 const discovery=await fetch('/api');assert.equal(discovery.status,200);assert.equal((await discovery.json()).openapi,'https://zer.foo/openapi.json');
 for(const [path,method,status] of [['/api/missing','GET',404],['/api','POST',405],['/docs/','POST',405],['/missing','GET',404]]){const r=await fetch(path,'application/json',method);assert.equal(r.status,status);const e=await r.json();for(const key of ['code','message','error','hint','docs'])assert.equal(typeof e[key],'string');}
});
test('HEAD omits body and legacy creation redirect and security headers remain intact',async()=>{
 const head=await fetch('/','text/markdown','HEAD');assert.equal(head.status,200);assert.equal(await head.text(),'');assert.match(head.headers.get('content-type'),/markdown/);
 const redirect=await fetch('/create/');assert.equal(redirect.status,301);assert.equal(redirect.headers.get('location'),'https://zer.foo/#message');assert.equal(redirect.headers.get('x-content-type-options'),'nosniff');
 const text=await fetch('/llms.txt');assert.match(text.headers.get('content-type'),/text\/markdown/);
});
import {readFile} from 'node:fs/promises';import {htmlToMarkdown} from '../scripts/build-agent-content.mjs';
test('Markdown generation retains meaningful page content, code and links without navigation or scripts',()=>{
 const text=htmlToMarkdown('<main><nav>Noise</nav><h1>Guide</h1><p>Use <a href="/docs/">docs</a>.</p><pre><code>zerfoo version</code></pre><script>secret()</script><button>Copy</button></main>');assert.match(text,/# Guide/);assert.match(text,/\[docs\]\(\/docs\/\)/);assert.match(text,/```[\s\S]*zerfoo version/);assert.doesNotMatch(text,/Noise|secret|Copy/);
});
test('OpenAPI operations, typed contracts and homepage identity metadata are discoverable',async()=>{
 const spec=JSON.parse(await readFile('static/openapi.json','utf8'));assert.equal(spec.openapi,'3.1.1');const ids=[];
 for(const path of Object.values(spec.paths))for(const [method,op] of Object.entries(path)){if(!['get','post'].includes(method))continue;assert.ok(op.description.length>20);assert.ok(op.operationId);ids.push(op.operationId);assert.ok(op.responses['200'].content);for(const param of op.parameters||[])assert.ok(param.schema.type);}
 assert.equal(new Set(ids).size,ids.length);assert.equal(spec.components.schemas.DesignRequest.properties.messages.maxItems,16);
 const home=await readFile('content/_index.html','utf8');const data=JSON.parse(home.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)[1]);const org=data['@graph'].find(x=>x['@type']==='Organization');assert.equal(org.contactPoint[0].email,'david@zer.foo');assert.equal(org.address.postalCode,'94114');assert.match(home,/og:image/);assert.match(home,/<title>Zerfoo ML framework/);
 for(const slug of ['contact','privacy','about']){const page=await readFile(`content/${slug}.html`,'utf8');const main=page.match(/<main[^>]*>([\s\S]*?)<\/main>/)[1].replace(/<[^>]*>/g,'');assert.ok(main.length>=500);}
 const llms=await readFile('static/llms.txt','utf8');assert.match(llms,/^# Zerfoo ML framework\n\n> /);assert.match(llms,/## When to use this/);for(const section of llms.split(/^## /m).slice(1))assert.match(section,/\n\n- \[.+\]\(https:\/\//);
});
import SwaggerParser from '@apidevtools/swagger-parser';
test('published document validates as OpenAPI 3.1',async()=>{await SwaggerParser.validate('static/openapi.json');});
