// vertex shader
vs = `
precision mediump float;
attribute vec3 position;
attribute vec3 normals;
attribute vec2 texcoord;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_depth;
uniform float u_scale;
uniform vec2 u_mouse;
uniform float u_seed;
varying vec3 normal;
varying vec3 FragPos;
varying vec2 v_texCoord;

//
// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
//

vec3 mod289(vec3 x){
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x){
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute(vec3 x){
  return mod289(((x*34.0)+1.0)*x);
}

float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
// First corner
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

// Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
    + i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

//---


void main(){
  normal = normals;

  vec2 uv = position.xz / u_resolution;
  vec3 newPos = position;

  float SCALEX = 20.0 * 1./u_scale;
  float SCALEY = 20.0 * 1./u_scale;

  float t = u_time * -0.00004;
  float dirX = 0.; //sin(t);//-t;
  float dirY = -t;

  float yPos = 0.;
  float noise =  (snoise( u_seed + vec2(uv.x * SCALEX + dirX, uv.y * SCALEY + dirY )) + 1.0) / 2.0;
  float noise2 = (snoise( u_seed + vec2(uv.x * SCALEX + dirX, uv.y * SCALEY + dirY ) * 5.) + 1.0) / 2.0;
  yPos = mix(noise, noise2, 0.08);
  //yPos = noise;

  newPos.y = yPos * u_depth; //sin(position.x * position.z * u_time * 0.000001) * 2.;
  FragPos = vec3(model * vec4(newPos, 1.0));
  v_texCoord = texcoord;
  gl_Position = projection * view * model * vec4(newPos, 1.0);
}
`;

//fragment shader
fs = `
// Author: Tobias Toft
// Title: Mountain ridges for Quarter Studio, v. 0.0.2
precision mediump float;
uniform mat4 view;
uniform vec3 u_lightPos;
uniform float u_ambientIntensity;
uniform float u_specularIntensity;
uniform vec3 u_ambientLightColor;
uniform vec3 u_lightColor;
uniform vec2 u_resolution;
uniform float u_time;
uniform sampler2D u_noise;
uniform float u_depth;
uniform vec2 u_mouse;
varying vec3 normal;
varying vec3 FragPos;
varying vec2 v_texCoord;


float whiteNoise(in vec2 p){
  vec2 tv = p * vec2(u_resolution.x/256., u_resolution.y/256.);
  return (texture2D(u_noise, tv).r);
}

vec3 hsv2rgb(vec3 c){
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float getDepth(float n, float cutoff, float steps){
  //remap remaining non-cutoff region to 0 - 1
  float d = (n - cutoff) / (1. - cutoff);

  //step
  d = floor(d*steps)/steps;
  return d;
}

float mapRange(float val, float low1, float high1, float low2, float high2){
  return low2 + (val - low1) * (high2 - low2) / (high1 - low1);
}

void main(){
  vec2 uv = gl_FragCoord.xy / u_resolution;
  float yPos = FragPos.y / u_depth; //y position in 0..1 range
  vec3 lightColor = u_lightColor;
  vec3 ambientLightColor = u_ambientLightColor;

  // Slice hills into layers
  const int steps = 40; //number of layers
  const int skip = 6; //for n layer, do something
  const int offset = 0; //start counting n layers with an offset

  // Calc pixel color
  float h = 0.68 + getDepth(yPos, 0.0, float(steps)) * 0.28;
  float s = 0.0;
  float v = 0.; //yPos; //floor(sin(yPos * 3.) + 0.2);

  float d = getDepth(yPos, 0.0, float(steps));
  const float thickness = 0.1; //for making simple gradients

  for (int i = 0; i<steps; i+=skip){
    float val = float(mod(float(i + offset), float(steps)))/float(steps);

    if (yPos <= val && yPos > val - thickness){
      v += mapRange(yPos, val - thickness, val, 0.0, 0.62); // gradient falloff
      //v = d;
      //s = 0.8;
    }

    // //Inverse
    // if (yPos <= val + thickness && yPos > val){
    //   v = mapRange(yPos, val + thickness, val, 0.0, 0.5);
    // }

    // // Thin solid line
    if (yPos <= val && yPos > val - 0.005){
      v = 1.;
      //s = 0.8;
    }
  }

  // Add noise and clamp values
  v *= floor((whiteNoise(uv) * v) + 0.5);
  v = clamp(v, 0.0, 1.0);

  // Add lighting
  vec3 ambient = u_ambientIntensity * ambientLightColor;
  vec3 lightPos = u_lightPos;

  vec3 norm = normalize(normal);
  vec3 lightDir = normalize(lightPos - FragPos);
  float diffImpact = max(dot(norm, lightDir), 0.0);
  vec3 diffuseLight = diffImpact * lightColor;

  vec3 viewPos = vec3(-u_mouse.x * u_resolution.x/5., -500.0, u_mouse.y * u_resolution.y/5.); //set light origin
  vec3 viewDir = normalize(viewPos - FragPos);
  vec3 reflectDir = reflect(-lightDir, norm);

  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.);
  vec3 specular = u_specularIntensity * spec * lightColor;

  vec3 lighting = diffuseLight + ambient + specular;

  // Set diffuse color
  vec3 diffuseColor = hsv2rgb(vec3(h, s, v));
  vec3 col = diffuseColor * lighting;


  // Add vertical gradient
  //col *= 0.05 + (gl_FragCoord.y/u_resolution.y) * 0.95;

  gl_FragColor = vec4(col, v);
}
`;

// get mouse position for GL uniform
function getRelativeMousePosition(event, target) {
  target = target || event.target;
  var rect = target.getBoundingClientRect();

  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  }
}

// assumes target or event.target is canvas
function getNoPaddingNoBorderCanvasRelativeMousePosition(event, target) {
  target = target || event.target;
  var pos = getRelativeMousePosition(event, target);

  pos.x = pos.x * target.width  / target.clientWidth;
  pos.y = pos.y * target.height / target.clientHeight;

  return pos;
}

// resize canvas
function resize(canvas) {
  // Lookup the size the browser is displaying the canvas.
  var displayWidth  = canvas.clientWidth;
  var displayHeight = canvas.clientHeight;

  // Check if the canvas is not the same size.
  if (canvas.width  !== displayWidth ||
      canvas.height !== displayHeight) {

    // Make the canvas the same size
    canvas.width  = displayWidth;
    canvas.height = displayHeight;
  }
}


// init webgl
let mousePos = {x: 0, y: 0};

function main(image){
  const m4 = twgl.m4;
  const gl = document.getElementById("c").getContext("webgl");
  const shader = twgl.createProgramInfo(gl, [vs, fs]);

  const vertices = 256;
  const offset = 0; //(vertices - 1) / 2;
  const strips = vertices - 1;
  const strip_length = vertices * 2;


  let position = [];
  let normal = [];
  let indices = [];

  const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
  for(let x = 0; x < vertices; x++) {
    for(let z = 0; z < vertices; z++) {
      //arrays.position.push(x - offset, ctx.getImageData(x, z, 1, 1).data[0] / z_compression, z - offset);
      position.push((x * aspect) - (vertices * aspect)/2, 0, z - vertices/2);
      normal.push(0, 0, 1);
    }
  }

  for (let z = 0; z < vertices - 1; z++) {
    if (z > 0) {
      indices.push(z * vertices);
    }
    for (let x = 0; x < vertices; x++) {
      indices.push((z * vertices) + x);
      indices.push(((z + 1) * vertices) + x);
    }

    if (z < vertices - 2) {
      indices.push(((z + 1) * vertices) + (vertices - 1));
    }
  }

  const arrays = {
    position: { numComponents: 3, data: position},
    normal: { numComponents: 3, data: normal},
    indices: { numComponents: 1, data: indices}
  };

  const buffers = twgl.createBufferInfoFromArrays(gl, arrays);

  const textures = twgl.createTextures(gl, {
    noise: { src: "js/noise.png", mag: gl.LINEAR }
  });

  const uniforms = {
    model: [],
    view: [],
    projection: [],
    u_lightPos: [0, 200, 0],
    u_lightColor: [1, 1, 1],//[0.5, 0.1, 0.0],
    u_ambientLightColor: [1, 1, 1],//[0.5, 0.4, 0.5],
    u_ambientIntensity: 0.2,
    u_specularIntensity: 0.5,
    u_time: 0,
    u_resolution: [gl.canvas.width, gl.canvas.height],
    u_noise: textures.noise,
    u_mouse: [mousePos.x, mousePos.y],
    u_depth: 70,
    u_scale: 1.0,
    u_seed: Math.random()
  };

  gl.clearColor(.0, .0, .0, 1);

  function render(time) {
    twgl.resizeCanvasToDisplaySize(gl.canvas, window.devicePixelRatio);

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.frontFace(gl.CW); // Apparently I draw triangles backwards
    gl.enable(gl.u_DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    //gl.clear(gl.COLOR_BUFFER_BIT | gl.u_DEPTH_BUFFER_BIT);
    //gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);
    //gl.enable(gl.BLEND);
    //gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    let zoom = 23;
    let fov = zoom * Math.PI / 130;
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const zNear = 0.1;
    const zFar = vertices * 2000;
    const projection = m4.perspective(fov, aspect, zNear, zFar);

    const eye = [0, 350, -120];
    const target = [0, 0, 0];
    const up = [0, 1, 0];
    const camera = m4.lookAt(eye, target, up);

    uniforms.view = m4.inverse(camera);
    uniforms.model = m4.identity();
    uniforms.projection = projection;
    uniforms.u_time = time;
    uniforms.u_mouse = [mousePos.x, mousePos.y];
    uniforms.u_resolution = [gl.canvas.width, gl.canvas.height];

    gl.useProgram(shader.program);

    twgl.setBuffersAndAttributes(gl, shader, buffers);
    twgl.setUniforms(shader, uniforms);
    //twgl.drawBufferInfo(gl, gl.TRIANGLES, buffers);

    gl.drawElements(gl.TRIANGLE_STRIP, buffers.numElements, gl.UNSIGNED_SHORT, 0);

    requestAnimationFrame(render);
  }

  window.addEventListener('mousemove', e => {
    const pos = getNoPaddingNoBorderCanvasRelativeMousePosition(e, gl.canvas);

    // pos is in pixel coordinates for the canvas.
    // so convert to WebGL clip space coordinates
    const x = pos.x / gl.canvas.width  *  2 - 1;
    const y = pos.y / gl.canvas.height * -2 + 1;

    mousePos.x = x;
    mousePos.y = y;
  });


  requestAnimationFrame(render);
}

/*
const image = new Image;
image.onload = () => {
  main(image);
}
image.src = 'js/noise.png'
*/

main();

/*
function render(time) {
  twgl.resizeCanvasToDisplaySize(gl.canvas, 1); //use window.devicePixelRatio if you want retina, we might not though...
  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

  const uniforms = {
    u_time: time * 0.001,
    u_resolution: [gl.canvas.width, gl.canvas.height],
    u_mouse: [mousePos.x, mousePos.y],
    u_texture: textures.noise,
    u_ashima1: textures.ashima1,
    u_ashima2: textures.ashima2
  };

  gl.useProgram(programInfo.program);
  twgl.setBuffersAndAttributes(gl, programInfo, bufferInfo);
  twgl.setUniforms(programInfo, uniforms);
  twgl.drawBufferInfo(gl, bufferInfo);

  requestAnimationFrame(render);
}

requestAnimationFrame(render);
*/

// attach listener for updating mouse position uniform



