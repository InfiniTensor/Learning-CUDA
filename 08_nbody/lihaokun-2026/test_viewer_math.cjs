// Pure JS logic tests with DOM/WebGL stubs, NOT a browser or visual test.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const {webcrypto} = require('node:crypto');
const html = fs.readFileSync(path.join(__dirname, 'trajectory_viewer.html'), 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const gl = new Proxy({COMPILE_STATUS: 1, LINK_STATUS: 2, POINTS: 0, LINES: 1, LINE_STRIP: 3,
  getShaderParameter:()=>true, getProgramParameter:()=>true, getAttribLocation:()=>1,
  getUniformLocation:()=>1, createProgram:()=>({}), createBuffer:()=>({}), createShader:()=>({})},
  {get:(target,key)=>key in target?target[key]:()=>{}});
const context2d = new Proxy({}, {get:()=>()=>{}, set:()=>true});
const elements = new Map();
for(const id of [...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1])){
  elements.set(id, {style:{},textContent:'',className:'',value:'',checked:false,width:1000,height:600,
    clientWidth:1000,clientHeight:600,getBoundingClientRect:()=>({width:300,height:170,left:0,top:0}),
    getContext:type=>type==='webgl'?gl:context2d,setPointerCapture:()=>{}});
}
elements.get('pointSize').value='2.5';elements.get('rate').value='25';
const ctx = vm.createContext({console,document:{getElementById:id=>elements.get(id)},window:{},
  devicePixelRatio:1,setTimeout,requestAnimationFrame:()=>{},crypto:webcrypto,ArrayBuffer,DataView,
  Uint8Array,Uint32Array,Float32Array,Float64Array});
vm.runInContext(code,ctx,{filename:'trajectory_viewer.html'});
function run(source){return vm.runInContext(source,ctx);}
function bin(n,r){const buffer=new ArrayBuffer(8+n*r*12),view=new DataView(buffer);view.setInt32(0,n,true);view.setInt32(4,r,true);return buffer;}
(async()=>{
  const buffer=bin(3,5),values=new Float32Array(buffer,8);
  for(let i=0;i<3;i++)for(let f=0;f<5;f++){const at=(i*5+f)*3;values.set([50000+i+f*.25,2*i+f*.5,3*i],at);}
  ctx.testBuffer=buffer;
  await run('loadBuffer(testBuffer, {name:"fixture.bin"})');
  assert.equal(run('S.N'),3);assert.equal(run('S.R'),5);
  run('selectParticle(1); S.frame=4; fitPath(); drawScene()');
  assert.equal(run('S.selected'),1);
  assert.ok(run('S.half')<3,'fit should show true small movement within a huge world');
  assert.match(elements.get('stats').textContent,/距初始位置 2.236068/);
  assert.match(elements.get('hud').textContent,/无位移放大/);
  run("S.mode='xy'; S.center=[50000,0,0]; S.half=10");
  assert.equal(run('project([50000,0,0])[0]'),500);
  assert.equal(run('project([50000,0,0])[1]'),300);
  run('selectParticle(999)');assert.equal(run('S.selected'),1);
  run('overview()');const original=run('S.half');
  elements.get('view').onwheel({deltaY:-200,preventDefault(){}});
  assert.ok(run('S.half')<original);
  elements.get('particle').value='2';elements.get('select').onclick();assert.equal(run('S.selected'),2);
  elements.get('frame').oninput({target:{value:'2'}});assert.equal(run('S.frame'),2);assert.equal(run('S.playing'),false);
  run('S.frame=4');elements.get('play').onclick();assert.equal(run('S.frame'),0);assert.equal(run('S.playing'),true);
  const hash=Buffer.from(await webcrypto.subtle.digest('SHA-256',buffer)).toString('hex');
  const report={schema:'nbody-validation-v1',trajectory_sha256:hash,status:'PASS_SAMPLED',scope:'test sampled scope'};
  await elements.get('report').onchange({target:{files:[{text:async()=>JSON.stringify(report)}]}});
  assert.match(elements.get('validation').textContent,/抽样回放通过/);
  assert.ok(!elements.get('validation').className.includes('ok'));
  report.trajectory_sha256='0'.repeat(64);
  await elements.get('report').onchange({target:{files:[{text:async()=>JSON.stringify(report)}]}});
  assert.match(elements.get('validation').textContent,/当前文件不同/);
  // Large particle count: no sampling of IDs; last ID remains selectable.
  const large=bin(65536,2);ctx.testBuffer=large;
  await run('loadBuffer(testBuffer)');run('selectParticle(65535); fitPath(); drawScene()');
  assert.equal(run('S.positions.length'),65536*3);assert.equal(run('S.selected'),65535);
  const malformed=bin(2,2);new DataView(malformed).setInt32(0,5,true);ctx.testBuffer=malformed;
  await assert.rejects(()=>run('loadBuffer(testBuffer)'),/BIN 尺寸不匹配/);
  console.log('PASS: BIN layout, 65536 IDs, local path geometry, zoom, selection, timeline, playback and SHA256 status logic.');
  console.log('DOM/WebGL were stubs: browser rendering and real GPU performance remain untested.');
})().catch(error=>{console.error(error);process.exitCode=1;});
