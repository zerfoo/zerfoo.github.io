// Visualize the public project contract; never execute model code.
export function adaptProject(value, ) {
 const exampleLabels = null;
 const p=value?.project ?? value;
 if(!p || p.version!==1 || !['numeric_classification','component_design','design_brief'].includes(p.task)) throw Error('Choose a version-1 Zerfoo project.json or designer response.');
 for(const [key,max] of [['objective',2400],['target',800],['architecture',2400]]) if(p[key]!=null&&(typeof p[key]!=='string'||p[key].length>max)) throw Error(`Invalid ${key}.`);
 if(!Array.isArray(p.features)||p.features.length>128||p.features.some(f=>typeof f!=='string'||!f.length||f.length>64)) throw Error('Expected up to 128 feature names.');
 if(p.components!=null&&(!Array.isArray(p.components)||p.components.length>16||p.components.some(c=>typeof c!=='string'||c.length>300))) throw Error('Invalid component list.');
 const supported=p.task==='numeric_classification'&&p.recipe==='dense-relu-16-v1';
 if(supported&&(!p.features.length||new Set(p.features).size!==p.features.length||p.features.includes(p.target))) throw Error('The numeric recipe needs distinct features that exclude the target.');
 let layers=supported?[
  {id:'input',name:'Your inputs',kind:'Input',count:p.features.length,color:'#78d0cd',subtitle:`${p.features.length} numeric features`,description:'One number for each measurement. These are the columns your model will read, in the order shown.',items:p.features,shape:`[batch, ${p.features.length}]`,params:0},
  {id:'dense',name:'Find patterns',kind:'Dense',count:16,color:'#e4d58f',subtitle:'16 connected units',description:'Each unit combines all your inputs. Training adjusts these connections to find patterns that help distinguish the classes.',items:['Every input connects to every unit','Weights and biases learned locally'],shape:'[batch, 16]',params:p.features.length*16+16},
  {id:'relu',name:'Add flexibility',kind:'ReLU',count:16,color:'#d4d77b',subtitle:'ReLU activation',description:'Keeps positive values and replaces negative values with zero. This lets the model learn more than a straight-line relationship.',items:['16 values in → 16 values out','No trainable parameters'],shape:'[batch, 16]',params:0},
  {id:'output',name:'Class scores',kind:'Dense',count:exampleLabels?.length??null,color:'#a4d5a8',subtitle:exampleLabels?`${exampleLabels.length} example classes`:'Class count determined locally',description:exampleLabels?'One score for each example class. These labels illustrate the design; no trained model or prediction is loaded.':'One score for each distinct target label. The number of classes is discovered when your local dataset is inspected.',items:exampleLabels??['Target: '+(p.target||'not specified'),'Class labels are not in the design response'],shape:`[batch, ${exampleLabels?.length??'classes'}]`,params:exampleLabels?17*exampleLabels.length:null}
 ]:(p.components||[]).map((id,i)=>({id:'component-'+i,name:id.split('/').pop(),kind:'Component',count:1,color:'#78d0cd',subtitle:'Proposed component',description:'Selected by the designer. Its dimensions, connections and implementation still need local validation.',items:[id],shape:'Not yet specified',params:null}));
 if(p.task==='component_design'&&p.topology!=null)layers=topologyLayers(p.topology);
 return {project:p,layers,supported:supported||!!(p.task==='component_design'&&p.topology),starter:supported,example:!!exampleLabels,objective:p.objective||'Your model design',parameters:supported&&exampleLabels?(p.features.length+1)*16+17*exampleLabels.length:null};
}

function topologyLayers(topology){
 if(topology.type!=='sequential'||!Array.isArray(topology.layers)||topology.layers.length<3||topology.layers.length>12)throw Error('Invalid sequential topology.');
 let previous=null;
 const colors=['#78d0cd','#e4d58f','#d4d77b','#a4d5a8'];
 const layers=topology.layers.map((layer,i)=>{
  const kinds=['Input','Dense','ReLU','Sigmoid','Tanh','Softmax','Dropout'];
  if(!layer||!kinds.includes(layer.kind)||(i===0)!==(layer.kind==='Input'))throw Error('Invalid topology layer.');
  const count=layer.units??null;
  if(count!==null&&(!Number.isSafeInteger(count)||count<1||count>4096))throw Error('Invalid topology units.');
  const activation=!['Input','Dense'].includes(layer.kind);
  if(activation&&count!==previous)throw Error('Activation must preserve layer width.');
  const params=layer.kind==='Dense'?(previous!==null&&count!==null?(previous+1)*count:null):0;
  previous=count;
  return {id:'layer-'+i,name:layer.kind==='Input'?'Input':layer.kind==='Dense'?(i===topology.layers.length-1?'Output layer':'Hidden layer') :layer.kind,kind:layer.kind,count,color:colors[i%colors.length],subtitle:count===null?'Width not yet specified':`${count} ${activation?'values':'units'}`,description:layer.kind==='Dense'?'A proposed fully connected layer. Its weights and biases will be learned during local training.':layer.kind==='Input'?'The model input. Its width will be set for your task.':'A proposed '+layer.kind+' operation, preserving the preceding layer width.',items:['Proposed sequential connection',count===null?'Dimension remains unspecified':`${count} values${count>32?' (32 drawn)':''}`],shape:`[batch, ${count??'?'}]`,params};
 });
 if(!layers.some(x=>x.kind==='Dense'))throw Error('Dense topology required.');
 return layers;
}
