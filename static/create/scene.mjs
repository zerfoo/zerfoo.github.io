import * as THREE from '../vendor/three/three.module.js';
import {OrbitControls} from '../vendor/three/OrbitControls.js';
export function createScene(host,onSelect) {
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
 renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0,0);host.append(renderer.domElement);
 renderer.domElement.setAttribute('aria-label','Interactive architecture. Use the layer buttons below for keyboard access.');
 const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(36,1,.1,120);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.enablePan=false;controls.minDistance=13;controls.maxDistance=40;controls.maxPolarAngle=Math.PI*.8;controls.minPolarAngle=.25;
 const ambient=new THREE.AmbientLight(0xffffff,2);scene.add(ambient);
 const key=new THREE.DirectionalLight(0xffffff,3);key.position.set(2,8,12);scene.add(key);
 const rim=new THREE.PointLight(0x78d0cd,80);rim.position.set(-5,1,4);scene.add(rim);
 let group=new THREE.Group();scene.add(group);let meshes=[],lines=[],particles=[],selected=1,dirty=true,flow=false,flowTime=0,lastTime=0,light=false,positions=[],model;
 const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
 const xPositions=[-5.7,-1.7,1.6,5.4];
 function reset(){camera.position.set(3.8,3.2,15.8);controls.target.set(0,0,0);controls.update();dirty=true;}
 function label(text,x,y,color){const canvas=document.createElement('canvas');canvas.width=640;canvas.height=80;const c=canvas.getContext('2d');c.font='500 28px Inter, Arial';c.textAlign='center';c.fillStyle=color;c.fillText(text,320,47);const texture=new THREE.CanvasTexture(canvas);const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false}));sprite.position.set(x,y,0);sprite.scale.set(4,.5,1);group.add(sprite);}
 function clear(){group.traverse(o=>{o.geometry?.dispose();for(const m of [].concat(o.material||[])){m.map?.dispose();m.dispose();}});scene.remove(group);group=new THREE.Group();scene.add(group);meshes=[];lines=[];particles=[];positions=[];}
 function build(next){model=next;clear();if(!next.supported){dirty=true;return;}
 const grid=new THREE.GridHelper(24,32,light?0xccccbf:0x363730,light?0xddddcf:0x24251f);grid.position.y=-3.3;grid.material.transparent=true;grid.material.opacity=.36;group.add(grid);
 next.layers.forEach((layer,index)=>{
  const count=Math.min(layer.count??1,32),arr=[];let color=new THREE.Color(layer.color);if(light)color.multiplyScalar(.66);
  const meshGroup=new THREE.Group();meshGroup.userData.layer=index;
  for(let n=0;n<count;n++){
   let y,z;if(index===1||index===2){y=(1.5-Math.floor(n/4))*1.05;z=(n%4-1.5)*1.05;}else{const cols=count>8?Math.ceil(count/8):1;const rows=Math.ceil(count/cols);y=((rows-1)/2-Math.floor(n/cols))*.88;z=(n%cols-(cols-1)/2)*.9;}
   const pos=new THREE.Vector3(xPositions[index],y,z);arr.push(pos);
   const sphere=new THREE.Mesh(new THREE.SphereGeometry(index===2?.085:.145,20,16),new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:light?.1:.4,metalness:.35,roughness:.28,transparent:true,opacity:index===2?.72:1}));sphere.position.copy(pos);sphere.userData.layer=index;meshGroup.add(sphere);meshes.push(sphere);
   if(index!==2){const halo=new THREE.Mesh(new THREE.SphereGeometry(.22,16,12),new THREE.MeshBasicMaterial({color,transparent:true,opacity:light?.07:.055,depthWrite:false}));halo.position.copy(pos);meshGroup.add(halo);}
  }
  const plateGeometry=new THREE.BoxGeometry(.025,4.4,index===0||index===3?2.1:4.3);
  const plate=new THREE.Mesh(plateGeometry,new THREE.MeshBasicMaterial({color,transparent:true,opacity:.035,depthWrite:false}));plate.position.set(xPositions[index],0,0);plate.userData.layer=index;meshGroup.add(plate);meshes.push(plate);
  const edge=new THREE.LineSegments(new THREE.EdgesGeometry(plateGeometry),new THREE.LineBasicMaterial({color,transparent:true,opacity:.22}));edge.position.copy(plate.position);meshGroup.add(edge);group.add(meshGroup);
  label(['INPUT','DENSE · 16','ReLU',layer.count===null?'OUTPUT · ?':'OUTPUT'][index],xPositions[index],-2.8,light?'#60625e':'#a5a5a5');
  positions.push(arr);
 });
 for(let l=0;l<3;l++)for(let a=0;a<positions[l].length;a++)for(let b=0;b<positions[l+1].length;b++){
  if(l===1&&a!==b)continue;
  const start=positions[l][a],end=positions[l+1][b],color=model.layers[l].color;
  const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([start,end]),new THREE.LineBasicMaterial({color:light?0x899483:color,transparent:true,opacity:l===1?.16:.095,depthWrite:false}));line.userData={from:l,to:l+1};group.add(line);lines.push(line);
  if((a+b)%3===0){const particle=new THREE.Mesh(new THREE.SphereGeometry(.045,8,6),new THREE.MeshBasicMaterial({color}));particle.visible=false;particle.userData={start,end,phase:(a+b*.37)%1,stage:l};group.add(particle);particles.push(particle);}
 }
 select(selected);dirty=true;
 }
 function select(index){selected=index;for(const mesh of meshes){if(mesh.material.emissive)mesh.material.emissiveIntensity=light?(mesh.userData.layer===index?.12:.03):(mesh.userData.layer===index?.65:.22);}
 for(const line of lines)line.material.opacity=line.userData.from===index||line.userData.to===index?(light?.5:.22):(light?.16:.045);dirty=true;}
 function resize(){const {width,height}=host.getBoundingClientRect();if(!width||!height)return;renderer.setSize(width,height,false);camera.aspect=width/height;camera.fov=width<500?49:36;camera.updateProjectionMatrix();dirty=true;}
 let down;renderer.domElement.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});
 renderer.domElement.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;const r=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);const hit=ray.intersectObjects(meshes)[0];if(hit)onSelect(hit.object.userData.layer);});
 controls.addEventListener('change',()=>dirty=true);
 const observer=new ResizeObserver(resize);observer.observe(host);reset();resize();
 renderer.setAnimationLoop(t=>{const dt=Math.min((t-lastTime)/1000,.05);lastTime=t;if(document.hidden)return;controls.update();if(flow){flowTime+=dt;for(const p of particles){const cycle=(flowTime*.45+p.userData.phase)%3;p.visible=Math.floor(cycle)===p.userData.stage;p.position.lerpVectors(p.userData.start,p.userData.end,cycle%1);}dirty=true;}if(dirty){renderer.render(scene,camera);dirty=false;}});
 renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();host.dispatchEvent(new Event('preview-unavailable'));});
 return {build,select,reset,setFlow(value){flow=value;for(const p of particles)p.visible=false;dirty=true;},setTheme(value){light=value;ambient.intensity=light?.8:2;key.intensity=light?1:3;if(model)build(model);},dispose(){observer.disconnect();controls.dispose();clear();renderer.setAnimationLoop(null);renderer.dispose();renderer.domElement.remove();}};
}
