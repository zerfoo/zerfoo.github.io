import {trackFunnel} from './analytics.mjs';
import {adaptProject} from './model.mjs?v=sequential-1';
import {createModelViewer} from './viewer.mjs?v=sequential-1';
const $=id=>document.getElementById(id);
let messages=[],project=null,download=null,busy=false,chatStarted=false;
const viewerRoot=document.querySelector?.('[data-model-viewer]');
const viewer=viewerRoot?createModelViewer(viewerRoot):null;
function append(role,content){const p=document.createElement('p');p.className=role;p.textContent=content;$('messages').append(p);const scroll=document.querySelector?.('.chat-scroll');if(scroll)scroll.scrollTop=scroll.scrollHeight;else p.scrollIntoView({block:'nearest'});}
function preview(p,model){project=p;viewer?.render(model);$('objective').textContent=p.objective;$('architecture').textContent=p.architecture||(model.supported?'Numeric inputs → Dense(16) → ReLU → Dense(class count determined locally)':'The architecture needs local engineering.');$('compute').textContent=[p.hardware,p.runtime].filter(Boolean).join(' · ')||'To be validated locally.';$('research').textContent=p.research_status||'No reviewed evidence attached.';for(const card of p.related_research||[]){if(typeof card.url!=='string'||!/^https:\/\/arxiv\.org\/abs\/[\w.-]+$/.test(card.url)||typeof card.title!=='string')continue;const a=document.createElement('a');a.href=card.url;a.textContent=card.title+' (unreviewed)';a.target='_blank';a.rel='noopener';$('research').append(document.createElement('br'),a);}}
document.querySelectorAll('[data-example]').forEach(b=>b.onclick=()=>{if(busy)return;$('message').value=b.dataset.example;$('message').focus();});
$('chat').onsubmit=async event=>{
 event.preventDefault();if(busy)return;const content=$('message').value.trim();if(!content)return;
 const nextMessages=[...messages,{role:'user',content}];const body=JSON.stringify({messages:nextMessages});
 if(content.length>1800||nextMessages.some(m=>m.content.length>2000)||nextMessages.length>16||new TextEncoder().encode(body).byteLength>8000){$('error').textContent='This conversation has reached its size limit. Download your last design, then start a new design or continue with your coding agent.';return;}
 busy=true;if(!chatStarted){chatStarted=true;trackFunnel('chat_started');}let failureCode='network';$('send').disabled=true;$('new-design').disabled=true;$('message').readOnly=true;$('error').textContent='';$('design-progress').textContent='Designing with you…';viewer?.setStatus(project?'Updating design…':'Understanding your idea…');append('user',content);$('message').value='';
 try{
  const response=await fetch('https://design.zer.foo/api/design',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body,signal:AbortSignal.timeout(110000)});
  failureCode=response.status===429?'rate_limited':response.status===503?'unavailable':response.status>=500?'provider_unavailable':'invalid_request';const data=await response.json();
  if(!response.ok){failureCode=data.code||failureCode;throw Error(data.error||'Design service unavailable');}
  failureCode='invalid_response';if(typeof data.message!=='string'||!data.message.trim()||data.message.length>4000)throw Error('The designer returned an incomplete response. Please try again.');
  const model=data.project?adaptProject(data.project):null;
  const nextDownload=model&&typeof data.download?.base64==='string'&&data.download.base64.length<=2000000?data.download:null;
  if(nextDownload){try{const bytes=atob(nextDownload.base64);if(!bytes.startsWith('PK'))throw Error();}catch{throw Error('The project download was incomplete. Please try again.');}}
  messages=[...nextMessages,{role:'assistant',content:data.message}];append('assistant',data.message);
  if(model){download=nextDownload;preview(data.project,model);$('download').disabled=!download;trackFunnel('design_generated',{task:data.project.task});}
  else viewer?.setStatus(project?'Last design · awaiting update':'Clarifying your idea');
 }catch(error){trackFunnel('design_failed',{reason:failureCode});$('message').value=content;$('error').textContent=error.name==='TimeoutError'?'The designer took too long to respond. Your message is restored; try again when ready.':error.message;viewer?.setStatus(project?'Last successful design':'Design unavailable · try again');}
 finally{busy=false;$('send').disabled=false;$('new-design').disabled=false;$('message').readOnly=false;$('design-progress').textContent='';}
};
$('download').onclick=()=>{if(!project||!download)return;const bytes=Uint8Array.from(atob(download.base64),c=>c.charCodeAt(0));const url=URL.createObjectURL(new Blob([bytes],{type:'application/zip'}));const a=document.createElement('a');a.href=url;a.download='zerfoo-project.zip';a.click();trackFunnel('project_downloaded',{task:project.task});setTimeout(()=>URL.revokeObjectURL(url),1000);$('instruction').textContent='Open my downloaded Zerfoo project, follow AGENTS.md, and use Kazi to build, train, evaluate and run it on my hardware.';};
$('new-design').onclick=()=>{if(busy)return;messages=[];project=null;download=null;chatStarted=false;$('messages').replaceChildren();$('message').value='';$('error').textContent='';$('download').disabled=true;$('architecture').textContent='Not specified yet.';$('compute').textContent='Training runs on your hardware.';$('research').textContent='No design yet.';viewer?.clear();$('instruction').textContent='Read https://zer.foo/start/ and help me create, train and run a Zerfoo model on my hardware using Kazi.';$('copied').textContent='';$('message').focus();};
$('copy').onclick=async()=>{try{await navigator.clipboard.writeText($('instruction').textContent);$('copied').textContent='Copied';}catch{$('copied').textContent='Select and copy the instruction above.';}};
