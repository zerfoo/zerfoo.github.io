import test from 'node:test';
import assert from 'node:assert/strict';
import worker,{validateProposal,DesignBudget,boundedJSON,searchResearch,reviewedResearch} from '../worker/index.mjs';
import {filesFor,zip} from '../static/create/bundle.mjs';
const input={message:'Ready for local validation',project:{task:'numeric_classification',objective:'Classify flowers',target:'species',features:['sepal_length','sepal_width','petal_length','petal_width'],hardware:'laptop'}};
test('retrieves actual library notes without promoting them to evidence',()=>{
 const cards=searchResearch('AutoTrain');assert.ok(cards.some(c=>c.id==='2410.15735'));assert.ok(cards.every(c=>c.review_status==='unreviewed'));assert.deepEqual(searchResearch('zzzzzzzzzzzz'),[]);
 assert.equal(reviewedResearch.length,1);assert.equal(reviewedResearch[0].id,'2410.15735');assert.equal(reviewedResearch[0].supports_runnable_recipe,false);
});
test('rejects target leakage and unsupported tasks',()=>{
 assert.throws(()=>validateProposal({...input,project:{...input.project,features:['species']}}));
 assert.throws(()=>validateProposal({...input,project:{...input.project,task:'language_model'}}));
 assert.throws(()=>validateProposal({...input,project:{...input.project,features:[]}}));
});
test('provider cannot inject executable recipe or research claims',()=>{
 const p=validateProposal({...input,project:{...input.project,recipe:'shell',evidence:['fake']}}).project;
 assert.equal(p.recipe,'dense-relu-16-v1');assert.deepEqual(p.evidence,[]);
 const f=filesFor(p);assert.ok(f['train.py'].includes("call('model" )===false);assert.ok(f['predict.py'].includes('model_predict'));
 assert.ok(!f['AGENTS.md'].includes('Classify flowers'));assert.ok(zip(f).size>2000);
});
test('unsupported briefs do not become executable models',()=>{
 const p=validateProposal({...input,project:{...input.project,task:'design_brief'}}).project;
 assert.equal(p.recipe,null);assert.equal(p.status,'requires_engineering');
 const recipe=JSON.parse(filesFor(p)['model.recipe.json']);assert.equal(recipe.runnable,false);assert.deepEqual(recipe.layers,[]);assert.equal(recipe.definition,null);
});
test('proposal strings reject control characters',()=>{assert.throws(()=>validateProposal({...input,message:'bad\u0000text'}));assert.throws(()=>validateProposal({...input,project:{...input.project,objective:'bad\u0001text'}}));});
test('bounded parser rejects oversized streaming payload',async()=>{
 await assert.rejects(boundedJSON(new Request('https://test',{method:'POST',body:'x'.repeat(12001)})));
});
test('no provider dispatch without configured budget and key',async()=>{
 const result=await worker.fetch(new Request('https://test/api/design',{method:'POST',headers:{Origin:'https://zer.foo'},body:'{}'}),{SITE_ORIGIN:'https://zer.foo'});
 assert.equal(result.status,503);
 const forbidden=await worker.fetch(new Request('https://test/api/design',{method:'POST',headers:{Origin:'https://evil.test'},body:'{}'}),{SITE_ORIGIN:'https://zer.foo'});
 assert.equal(forbidden.status,403);
});
test('durable budget refuses overspend and repeated rapid requests',async()=>{
 const records=new Map();let chain=Promise.resolve();
 const store={get:async k=>records.get(k),put:async(k,v)=>records.set(k,v)};
 const ctx={storage:{transaction:f=>{const next=chain.then(()=>f(store));chain=next.catch(()=>{});return next;}}};
 const budget=new DesignBudget(ctx,{LIFETIME_BUDGET_CENTS:'1'});
 const req=id=>new Request('https://budget',{method:'POST',body:JSON.stringify({identity:id.repeat(64)})});
 const results=await Promise.all([budget.fetch(req('a')),budget.fetch(req('b'))]);
 assert.deepEqual(results.map(x=>x.status).sort(),[200,429]);assert.equal(records.get('reserved'),1);
});
