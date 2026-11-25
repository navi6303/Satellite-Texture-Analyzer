// Exercise the shipped canvas handlers, including native-pixel mapping at fit zoom.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const html = fs.readFileSync(path.join(__dirname, "../components/snip_canvas/index.html"), "utf8");
const handlers = {}, nodes = {}, messages = [];
const canvasContext = new Proxy({}, { get: (_, k) => k === "measureText" ? () => ({width:40}) : () => {} });
function node(id) {
  return nodes[id] ??= {
    clientWidth:960, clientHeight:560, style:{}, handlers:{}, classList:{add(){},remove(){}},
    getContext:()=>canvasContext, getBoundingClientRect:()=>({left:15,top:25}),
    addEventListener(type,fn){this.handlers[type]=fn;}, setPointerCapture(){}
  };
}
const context = {
  window:{parent:{postMessage(m){messages.push(m);}},addEventListener(t,fn){handlers[t]=fn;}},
  document:{getElementById:node,body:{scrollHeight:680}},
  Image:class {constructor(){this.naturalWidth=900;this.naturalHeight=600;}set src(v){this.url=v;if(v==="patch-preview"){this.naturalWidth=700;this.naturalHeight=520;}this.onload();}get src(){return this.url;}},
};
vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
handlers.message({data:{type:"streamlit:render",args:{category:"Image subset",image_data_url:"preview",orig_w:9000,orig_h:6000}}});
let zoom=.96*560/600, px=(960-900*zoom)/2, py=(560-600*zoom)/2;
function event(type,x,y) {
  node("sel").handlers[type]({clientX:15+px+x*zoom, clientY:25+py+y*zoom,
                              button:0, pointerId:1, preventDefault(){}});
}
function drag(x1,y1,x2,y2) {
  event("pointerdown",x1,y1);event("pointermove",x2,y2);event("pointerup",x2,y2);
}
drag(300,230,100,100); // reversed drawing
drag(300,230,340,260); // resize bottom-right
drag(180,160,200,170); // reposition
node("btnConfirm").onclick();
const value=messages.filter(m=>m.type==="streamlit:setComponentValue").at(-1).value;
assert.deepEqual(JSON.parse(JSON.stringify(value)),{x:1200,y:1100,w:2400,h:1600,confirmed:true,category:"Image subset"});
console.log("PASS: draw, resize, move and confirm return the expected native rectangle from a scaled preview.");

// Apply the native-size preview and CSS fit zoom of the real 700 × 520 test subset.
handlers.message({data:{type:"streamlit:render",args:{category:"Urban",image_data_url:"patch-preview",orig_w:700,orig_h:520}}});
zoom=.96*Math.min(960/700,560/520); px=(960-700*zoom)/2; py=(560-520*zoom)/2;
drag(484,328,472,320);
assert.match(node("coordDims").textContent, /12 × 8 native pixels \(96 total\)/);
node("btnConfirm").onclick();
const texture = messages.filter(m => m.type === "streamlit:setComponentValue").at(-1).value;
assert.deepEqual(JSON.parse(JSON.stringify(texture)), {x:472,y:320,w:12,h:8,confirmed:true,category:"Urban"});
console.log("PASS: the scaled texture canvas returns the 12 × 8 subset box at (472, 320).");
