import {readdir,readFile,writeFile} from 'node:fs/promises';
import {join,relative} from 'node:path';
import TurndownService from 'turndown';
import {gfm} from 'turndown-plugin-gfm';
export function htmlToMarkdown(html){
 const main=html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i)?.[1];
 if(!main)throw Error('Page has no main content');
 const service=new TurndownService({headingStyle:'atx',codeBlockStyle:'fenced',bulletListMarker:'-'});service.use(gfm);service.remove(['script','style','nav','aside','header','footer','button','label','svg','form','dialog']);
 service.addRule('heading-anchor',{filter:node=>node.nodeName==='A'&&node.textContent==='#'&&node.getAttribute('href')?.startsWith('#'),replacement:()=>''});
 const text=service.turndown(main).trim();
 return text+(text.length<20?'\n\nExplore [Zerfoo documentation](https://zer.foo/docs/) or [agent resources](https://zer.foo/llms.txt).':'')+'\n';
}
export async function build(root='public'){
 const paths=[];
 async function walk(dir){for(const item of await readdir(dir,{withFileTypes:true})){const path=join(dir,item.name);if(item.isDirectory())await walk(path);else if(item.name==='index.html')paths.push(path);}}
 await walk(root);
 for(const path of paths){const rel=relative(root,path),output=path.replace(/index\.html$/,'index.md');
  if(rel==='index.html'){await writeFile(output,await readFile('static/agent-home.md','utf8'));continue;}
  const html=await readFile(path,'utf8');if(!/<main\b/i.test(html))continue;
  await writeFile(output,htmlToMarkdown(html));
 }
 return paths.length;
}
if(process.argv[1]?.endsWith('build-agent-content.mjs'))console.log(`Prepared Markdown for ${await build()} pages.`);
