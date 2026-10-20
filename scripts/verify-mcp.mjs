import assert from 'node:assert/strict';import {writeFile} from 'node:fs/promises';
import {Client,StreamableHTTPClientTransport} from '@modelcontextprotocol/client';
import {handoffInstruction} from '../static/create/handoff.mjs';
const base=process.argv[2]||'https://zer.foo',url=new URL('/mcp',base),checks=[];
const client=new Client({name:'zerfoo-protocol-verifier',version:'1.0.0'},{versionNegotiation:{mode:{pin:'2026-07-28'}}});
try{
 await client.connect(new StreamableHTTPClientTransport(url));assert.equal(client.getProtocolEra(),'modern');checks.push('Official current MCP client connected using2026-07-28');
 const tools=await client.listTools();assert.deepEqual(tools.tools.map(x=>x.name),['search_docs','read_document','get_local_workflow']);checks.push('Three read-only tools discovered');
 const result=await client.callTool({name:'search_docs',arguments:{query:'inference',limit:3}});const found=result.structuredContent||result;assert.ok(found.results.length>0);
 const read=await client.callTool({name:'read_document',arguments:{url:found.results[0].url,length:256}});assert.ok((read.structuredContent||read).text.length>0);checks.push('Real published documentation searched and read');
 const denied=await client.callTool({name:'read_document',arguments:{url:'https://example.com/private'}});assert.equal(denied.isError,true);checks.push('Arbitrary URL denied');
 const resources=await client.listResources();assert.equal(resources.resources.length,3);const llms=await client.readResource({uri:'https://zer.foo/llms.txt'});assert.match(llms.contents[0].text,/Zerfoo/);checks.push('Reference resources read');
 const workflow=await client.callTool({name:'get_local_workflow',arguments:{}});assert.equal((workflow.structuredContent||workflow).instruction,handoffInstruction);const prompt=await client.getPrompt({name:'create_zerfoo_model'});assert.match(prompt.messages[0].content.text,/llms.txt/);checks.push('Canonical local workflow and prompt returned');
}finally{await client.close();}
async function legacy(body){const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json',Accept:'application/json, text/event-stream','MCP-Protocol-Version':'2025-11-25'},body:JSON.stringify(body)});const text=await response.text();const json=response.headers.get('content-type')?.includes('text/event-stream')?text.split('\n').filter(line=>line.startsWith('data: ')).map(line=>JSON.parse(line.slice(6))).at(-1):JSON.parse(text);assert.equal(response.status,200);return json;}
const init=await legacy({jsonrpc:'2.0',id:1,method:'initialize',params:{protocolVersion:'2025-11-25',capabilities:{},clientInfo:{name:'legacy-verifier',version:'1'}}});assert.equal(init.result.serverInfo.name,'zerfoo-docs');const listed=await legacy({jsonrpc:'2.0',id:2,method:'tools/list'});assert.equal(listed.result.tools.length,3);checks.push('Legacy initialization and discovery passed');
const foreign=await fetch(url,{method:'POST',headers:{Origin:'https://invalid.example','Content-Type':'application/json',Accept:'application/json, text/event-stream'},body:'{}'});assert.equal(foreign.status,403);checks.push('Foreign Origin rejected');
await writeFile('/tmp/zerfoo-mcp-verification.json',JSON.stringify({endpoint:url.href,checks},null,2)+'\n');console.log(`Verified ${checks.length} MCP checks at ${url.href}.`);
