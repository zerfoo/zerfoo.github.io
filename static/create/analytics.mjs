const events=new Set(['chat_started','design_generated','project_downloaded','design_failed']);
const tasks=new Set(['numeric_classification','component_design','design_brief']);
const reasons=new Set(['network','rate_limited','unavailable','provider_unavailable','provider_timeout','invalid_request','invalid_response','invalid_design']);
// Fixed categories only: never pass conversation, project text or provider errors.
export function trackFunnel(event,details={}) {
 if(!events.has(event))return;
 const params={};
 if(event==='design_generated'||event==='project_downloaded')params.design_task=tasks.has(details.task)?details.task:'unknown';
 if(event==='design_failed')params.failure_reason=reasons.has(details.reason)?details.reason:'unknown';
 try{if(typeof globalThis.gtag==='function')globalThis.gtag('event',event,params);}catch{/* Analytics must never interrupt creation. */}
}
