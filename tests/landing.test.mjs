import test from 'node:test';import assert from 'node:assert/strict';import {readFile,readdir} from 'node:fs/promises';
test('landing introduces the work and routes visitors into a separate working designer',async()=>{
 const home=await readFile('content/_index.html','utf8'),designer=await readFile('content/create.html','utf8');
 assert.match(home,/Build AI\.<br><span>Keep it yours\./);assert.match(home,/href="\/create\/"/);assert.doesNotMatch(home,/id="chat"|src="\/create\/app/);
 for(const id of ['chat','message','send','canvas-host','messages','new-design','download'])assert.ok(designer.includes(`id="${id}"`),id);
 assert.match(designer,/src="\/create\/app.mjs/);assert.match(designer,/rel="canonical" href="https:\/\/zer.foo\/create\/"/);assert.doesNotMatch(designer,/http-equiv="refresh"|<script>[^<]/);
 assert.match(home,/January 23 · Therfoo/);assert.match(home,/github.com\/therfoo\/therfoo\/commit\/24a3979/);assert.match(home,/Illustrative architecture/);assert.match(home,/not trained weights/);assert.doesNotMatch(home,/late on rent|Basis Fellowship|grant|broke/i);
});
test('navigation reaches the designer from each informational page',async()=>{
 for(const name of await readdir('content')){if(!name.endsWith('.html')||name==='create.html')continue;const html=await readFile(`content/${name}`,'utf8');assert.match(html,/href="\/create\/(?:#message)?"/,name);assert.doesNotMatch(html,/href="\/#message"/,name);}
 const agent=await readFile('static/agent-home.md','utf8');assert.match(agent,/https:\/\/zer.foo\/create\//);const llms=await readFile('static/llms.txt','utf8');assert.match(llms,/https:\/\/zer.foo\/create\//);
});
