import test from 'node:test';import assert from 'node:assert/strict';import {readFile,readdir} from 'node:fs/promises';
test('landing introduces the work and routes visitors into a separate working designer',async()=>{
 const home=await readFile('content/_index.html','utf8'),designer=await readFile('content/create.html','utf8');
 assert.match(home,/Build AI\.<br><span>Keep it yours\./);assert.match(home,/href="\/create\/"/);assert.doesNotMatch(home,/id="chat"|src="\/create\/app/);
 for(const id of ['chat','message','send','canvas-host','messages','new-design','download'])assert.ok(designer.includes(`id="${id}"`),id);
 assert.match(designer,/src="\/create\/app.mjs/);assert.match(designer,/rel="canonical" href="https:\/\/zer.foo\/create\/"/);assert.doesNotMatch(designer,/http-equiv="refresh"|<script>[^<]/);
 assert.match(home,/January 23 · Therfoo/);assert.match(home,/github.com\/therfoo\/therfoo\/commit\/24a3979/);assert.match(home,/GOOGLE GEMMA 3/);assert.match(home,/62 transformer decoder layers/);assert.doesNotMatch(home,/Illustrative multilayer perceptron|network-nodes/);assert.match(home,/not trained weights/);assert.doesNotMatch(home,/late on rent|Basis Fellowship|grant|broke/i);
});
test('navigation reaches the designer from each informational page',async()=>{
 for(const name of await readdir('content')){if(!name.endsWith('.html')||name==='create.html')continue;const html=await readFile(`content/${name}`,'utf8');assert.match(html,/href="\/create\/(?:#message)?"/,name);assert.doesNotMatch(html,/href="\/#message"/,name);}
 const agent=await readFile('static/agent-home.md','utf8');assert.match(agent,/https:\/\/zer.foo\/create\//);const llms=await readFile('static/llms.txt','utf8');assert.match(llms,/https:\/\/zer.foo\/create\//);
});

test('Gemma 3 uses complete responsive architecture assets and a full-width layout',async()=>{
 const home=await readFile('content/_index.html','utf8'),css=await readFile('static/css/landing.css','utf8');assert.match(home,/Google’s architecture guide/);assert.match(css,/\.landing-hero\{display:block\}/);assert.match(css,/\.gemma-visual\{width:100%;max-width:none/);
 for(const name of ['gemma3-architecture.svg','gemma3-architecture-mobile.svg']){const svg=await readFile(`static/images/${name}`,'utf8');for(const term of ['SigLIP encoder','Projector','Token embedding','RMSNorm','LM head','62','256 soft tokens'])assert.ok(svg.includes(term),`${name}: ${term}`);assert.equal((svg.match(/>LOCAL<|>L</g)||[]).length,5);assert.equal((svg.match(/>GLOBAL<|>G</g)||[]).length,1);}
});
