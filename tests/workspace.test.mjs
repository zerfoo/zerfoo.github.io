import test from 'node:test';
import assert from 'node:assert/strict';
import {adaptProject} from '../static/create/model.mjs';
const project={version:1,task:'numeric_classification',recipe:'dense-relu-16-v1',objective:'Classify flowers',features:['length','width'],target:'species',hardware:'CPU',runtime:'cpu/float32'};
test('actual numeric projects retain unknown output dimensions and exact feature names',()=>{const m=adaptProject(project);assert.deepEqual(m.layers[0].items,['length','width']);assert.equal(m.layers[1].count,16);assert.equal(m.layers[1].params,48);assert.equal(m.layers[3].count,null);assert.equal(m.parameters,null);assert.equal(m.example,false);});
test('component designs and briefs do not invent graph edges or dimensions',()=>{const m=adaptProject({...project,task:'component_design',recipe:null,components:['layers/attention:NewAttention'],architecture:'Attend over the input.'});assert.equal(m.supported,false);assert.equal(m.layers[0].shape,'Not yet specified');assert.equal(m.layers[0].params,null);assert.deepEqual(adaptProject({...project,task:'design_brief',recipe:null}).layers,[]);});
test('malformed projects and unsupported recipes cannot become the classifier',()=>{assert.throws(()=>adaptProject({...project,features:['species']}));assert.throws(()=>adaptProject({...project,features:['a','a']}));assert.throws(()=>adaptProject({...project,features:Array(129).fill('a')}));assert.throws(()=>adaptProject({version:2}));assert.equal(adaptProject({...project,recipe:'different'}).supported,false);});
test('real chat handlers keep successful history and downloads across clarification, failures, and reset',async()=>{
 const nodes=new Map();const node=()=>({value:'',textContent:'',hidden:false,disabled:false,children:[],append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},scrollIntoView(){},click(){},focus(){}});
 const oldDocument=globalThis.document,oldFetch=globalThis.fetch;
 globalThis.document={getElementById:id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);},createElement:node,querySelectorAll:()=>[]};
 const el=id=>document.getElementById(id),requests=[];let result;
 globalThis.fetch=async(url,options)=>{requests.push({url,options,body:JSON.parse(options.body)});if(result instanceof Error)throw result;return result;};
 const submit=async(text)=>{el('message').value=text;await el('chat').onsubmit({preventDefault(){}});};
 try{
  await import('../static/create/app.mjs?workspace-test');
  result=Response.json({message:'What is the target?',project:null});await submit('Classify flowers');assert.equal(el('objective').textContent,'');assert.equal(requests[0].options.credentials,'include');
  result=Response.json({message:'Here is the design.',project,download:{base64:'UEsDBA=='}});await submit('species');assert.equal(el('objective').textContent,project.objective);assert.equal(el('download').disabled,false);assert.equal(requests[1].body.messages.length,3);
  result=Response.json({error:'Please wait',code:'rate_limited'},{status:429});await submit('Add another feature');assert.equal(el('message').value,'Add another feature');assert.equal(el('objective').textContent,project.objective);assert.equal(el('send').disabled,false);assert.equal(el('download').disabled,false);
  result=Response.json({message:'Which feature?',project:null});await submit('Add another feature');assert.equal(requests.at(-1).body.messages.length,5);assert.equal(el('objective').textContent,project.objective);
  result=Response.json({message:'Invalid',project:{version:99}});await submit('bad');assert.match(el('error').textContent,/version-1/);assert.equal(el('objective').textContent,project.objective);
  el('new-design').onclick();assert.equal(el('download').disabled,true);assert.equal(el('messages').children.length,0);
  result=Response.json({message:'Starting fresh.',project:null});await submit('New problem');assert.equal(requests.at(-1).body.messages.length,1);
  const count=requests.length;await submit('x'.repeat(1801));assert.equal(requests.length,count);assert.match(el('error').textContent,/size limit/);
 }finally{globalThis.document=oldDocument;globalThis.fetch=oldFetch;}
});
