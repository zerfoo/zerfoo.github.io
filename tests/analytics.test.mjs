import test from 'node:test';
import assert from 'node:assert/strict';
import {trackFunnel} from '../static/create/analytics.mjs';
test('funnel analytics allows only fixed categories and survives blocked analytics',()=>{
 const calls=[];globalThis.gtag=(...args)=>calls.push(args);
 trackFunnel('chat_started',{message:'private chat'});
 trackFunnel('design_generated',{task:'numeric_classification',objective:'private project'});
 trackFunnel('project_downloaded',{task:'private text'});
 trackFunnel('design_failed',{reason:'private provider error'});
 trackFunnel('private event',{message:'private'});
 assert.deepEqual(calls,[['event','chat_started',{}],['event','design_generated',{design_task:'numeric_classification'}],['event','project_downloaded',{design_task:'unknown'}],['event','design_failed',{failure_reason:'unknown'}]]);
 globalThis.gtag=()=>{throw Error('blocked');};assert.doesNotThrow(()=>trackFunnel('chat_started'));
 delete globalThis.gtag;assert.doesNotThrow(()=>trackFunnel('chat_started'));
});
