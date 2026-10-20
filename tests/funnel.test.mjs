import test from 'node:test';
import assert from 'node:assert/strict';
test('real chat handlers emit one start, successful designs, failures and download initiation',async()=>{
 const nodes=new Map();const node=()=>({value:'',textContent:'',hidden:true,disabled:false,append(){},scrollIntoView(){},click(){},focus(){}});
 globalThis.document={getElementById:id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);},createElement:node,querySelectorAll:()=>[]};
 const events=[];globalThis.gtag=(...args)=>events.push(args);
 const originalFetch=globalThis.fetch;
 try{
  await import('../static/create/app.mjs');
  const submit=async()=>{document.getElementById('message').value='Private prompt';await document.getElementById('chat').onsubmit({preventDefault(){}});};
  globalThis.fetch=async()=>Response.json({error:'Private provider detail',code:'invalid_design'},{status:502});
  await submit();assert.equal(document.getElementById('message').value,'Private prompt');
  globalThis.fetch=async()=>Response.json({message:'Private response',project:{version:1,task:'numeric_classification',objective:'Private objective',features:['measurement'],target:'label',hardware:'CPU',runtime:'cpu',recipe:'dense-relu-16-v1'},download:{base64:'UEsDBA=='}});
  await submit();document.getElementById('download').onclick();
  assert.deepEqual(events.map(e=>e[1]),['chat_started','design_failed','design_generated','project_downloaded']);
  assert.doesNotMatch(JSON.stringify(events),/Private/);
  assert.equal(document.getElementById('send').disabled,false);
 }finally{globalThis.fetch=originalFetch;delete globalThis.gtag;delete globalThis.document;}
});
