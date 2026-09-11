import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync,readFileSync} from 'node:fs';
test('website contains UI, not private model creation implementation',()=>{
 for(const file of ['worker/index.mjs','worker/research.json','worker/reviewed-research.json','static/create/bundle.mjs','wrangler.jsonc'])assert.equal(existsSync(file),false,file);
 const client=readFileSync('static/create/app.mjs','utf8');assert.doesNotMatch(client,/filesFor|const TRAIN|SYSTEM=/);assert.match(client,/download.base64/);
 assert.doesNotMatch(readFileSync('static/start/index.html','utf8'),/go build|bfbb707/);
});
