import {writeFile,mkdir} from 'node:fs/promises';
import {filesFor,zip} from '../static/create/bundle.mjs';
import {validateProposal} from '../worker/index.mjs';
const project=validateProposal({message:'Ready',project:{task:'numeric_classification',objective:'Classify Iris flowers',target:'species',features:['sepal_length','sepal_width','petal_length','petal_width'],hardware:'CPU'}}).project;
const target=process.argv[2];if(!target)throw Error('Output directory required');
await mkdir(target,{recursive:true});
for(const [name,content] of Object.entries(filesFor(project)))await writeFile(target+'/'+name,content);
await writeFile(target+'/project.zip',new Uint8Array(await zip(filesFor(project)).arrayBuffer()));
