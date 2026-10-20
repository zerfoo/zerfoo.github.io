import test from 'node:test';import assert from 'node:assert/strict';
import {Client,StreamableHTTPClientTransport} from '@modelcontextprotocol/client';
import worker from '../worker/site.mjs';import {handoffInstruction} from '../static/create/handoff.mjs';
const page={url:'https://zer.foo/docs/example/index.md',path:'/docs/example/index.md',title:'Zerfoo inference and training',summary:'Run a GGUF model locally and evaluate on your hardware.',headings:'Inference Local training'};
const assets=new Map([['/agent-docs.json',JSON.stringify({version:1,pages:[page]})],['/llms.txt','# Zerfoo\n\nRead the agent instructions for local model creation.'],['/agent-instructions.txt','# Instructions\n\nUse installed Kazi interfaces.'],['/openapi.json','{"openapi":"3.1.1"}'],[page.path,'# Local inference\n\n'+('A public guide to local models. '.repeat(600))]]);
const calls=[];const env={ASSETS:{fetch:async req=>{const path=new URL(req.url).pathname;calls.push(path);return new Response(assets.get(path)||'Missing',{status:assets.has(path)?200:404,headers:{'Content-Type':'text/plain'}});}}};
const fetch=(input,init)=>worker.fetch(new Request(input,init),env);
async function withClient(run){const client=new Client({name:'zerfoo-test',version:'1.0.0'},{versionNegotiation:{mode:{pin:'2026-07-28'}}});const transport=new StreamableHTTPClientTransport(new URL('https://zer.foo/mcp'),{fetch});try{await client.connect(transport);assert.equal(client.getProtocolEra(),'modern');await run(client);}finally{await client.close();}}
test('official modern MCP client discovers tools, reads resources, searches and pages documentation',async()=>{
 await withClient(async client=>{
  const tools=await client.listTools();assert.deepEqual(tools.tools.map(x=>x.name),['search_docs','read_document','get_local_workflow']);for(const tool of tools.tools)assert.equal(tool.annotations.readOnlyHint,true);
  const resources=await client.listResources();assert.equal(resources.resources.length,3);const resource=await client.readResource({uri:'https://zer.foo/llms.txt'});assert.match(resource.contents[0].text,/# Zerfoo/);
  const found=await client.callTool({name:'search_docs',arguments:{query:'GGUF',limit:3}});const data=found.structuredContent||found;assert.equal(data.results[0].url,page.url);
  const read=await client.callTool({name:'read_document',arguments:{url:page.url,length:256}});const chunk=read.structuredContent||read;assert.equal(chunk.text.length,256);assert.equal(chunk.nextOffset,256);
  const workflow=await client.callTool({name:'get_local_workflow',arguments:{}});assert.equal((workflow.structuredContent||workflow).instruction,handoffInstruction);
  const prompt=await client.getPrompt({name:'create_zerfoo_model'});assert.match(prompt.messages[0].content.text,/llms.txt/);assert.match(prompt.messages[0].content.text,/Kazi/);
 });
});
test('MCP denies arbitrary documents and invalid tool arguments without fetching external data',async()=>{
 await withClient(async client=>{const count=calls.length;const denied=await client.callTool({name:'read_document',arguments:{url:'https://example.com/private'}});assert.equal(denied.isError,true);assert.equal(calls.length,count);const invalid=await client.callTool({name:'search_docs',arguments:{query:'x'.repeat(201)}});assert.equal(invalid.isError,true);});
});
function rpc(body,headers={}){return worker.fetch(new Request('https://zer.foo/mcp',{method:'POST',headers:{'Content-Type':'application/json',Accept:'application/json, text/event-stream',...headers},body:JSON.stringify(body)}),env);}
async function bodyOf(response){const text=await response.text();if(response.headers.get('content-type')?.includes('text/event-stream')){const data=text.split('\n').filter(x=>x.startsWith('data: ')).map(x=>JSON.parse(x.slice(6)));return data.at(-1);}return JSON.parse(text);}
test('legacy Streamable HTTP initialization and discovery work alongside modern clients',async()=>{
 const init=await rpc({jsonrpc:'2.0',id:1,method:'initialize',params:{protocolVersion:'2025-11-25',capabilities:{},clientInfo:{name:'legacy-test',version:'1'}}});assert.equal(init.status,200);assert.equal((await bodyOf(init)).result.serverInfo.name,'zerfoo-docs');
 const notice=await rpc({jsonrpc:'2.0',method:'notifications/initialized'},{'MCP-Protocol-Version':'2025-11-25'});assert.equal(notice.status,202);
 const listed=await rpc({jsonrpc:'2.0',id:2,method:'tools/list'},{'MCP-Protocol-Version':'2025-11-25'});assert.equal(listed.status,200);assert.equal((await bodyOf(listed)).result.tools.length,3);
});
test('MCP rejects foreign origins, oversized payloads, malformed JSON and unsupported methods',async()=>{
 assert.equal((await rpc({jsonrpc:'2.0',id:1,method:'tools/list'},{Origin:'https://evil.example'})).status,403);
 const large=await rpc({jsonrpc:'2.0',id:1,method:'tools/list',params:{padding:'x'.repeat(17000)}});assert.equal(large.status,413);
 const malformed=await worker.fetch(new Request('https://zer.foo/mcp',{method:'POST',headers:{'Content-Type':'application/json',Accept:'application/json, text/event-stream'},body:'{'}),env);assert.equal(malformed.status,400);
 const get=await worker.fetch(new Request('https://zer.foo/mcp',{headers:{Accept:'text/event-stream'}}),env);assert.equal(get.status,405);
});
test('website handoff text and reset use the canonical llms entry point',async()=>{
 const {readFile}=await import('node:fs/promises');for(const path of ['content/_index.html','content/create.html','content/how-it-works.html']){const html=await readFile(path,'utf8');assert.ok(html.includes(handoffInstruction));assert.doesNotMatch(html,/Read https:\/\/zer.foo\/start\//);}
 assert.match(handoffInstruction,/create, train, evaluate and run/);assert.match(handoffInstruction,/Kazi/);
});
