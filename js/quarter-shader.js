// vertex shader
vs = `
precision highp float;
attribute mediump vec3 position;
attribute mediump vec3 normals;
attribute mediump vec2 texcoord;
uniform mediump mat4 model;
uniform mediump mat4 view;
uniform mediump mat4 projection;
uniform mediump float u_time;
uniform mediump vec2 u_resolution;
uniform mediump vec2 u_direction;
uniform mediump float u_depth;
uniform mediump float u_scale;
uniform mediump float u_seed;
varying mediump vec3 v_normal;
varying mediump vec3 v_fragPos;
varying mediump vec2 v_texCoord;


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
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = i - floor(i * (1.0 / 289.0)) * 289.0; // Avoid truncation effects in permutation
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

//--- End of Ashima Simplex noise


float sinNoise(float t){
  return sin( (position.x * position.z * 0.0001) + t * 10. );
}

void main(){
  v_normal = normals;
  float aspect = u_resolution.x/u_resolution.y;
  vec2 uv = position.xz / u_resolution;
  vec4 newPos = vec4(position, 1.0);

  float SCALEX = 5.0 * aspect * 1./u_scale;
  float SCALEY = 5.0 * 1./u_scale;

  float dirX = u_direction.x * u_time; //sin(t);//-t;
  float dirY = u_direction.y * u_time;

  float noise1 = snoise( u_seed + vec2(uv.x * SCALEX + dirX, uv.y * SCALEY + dirY ));
  float noise2 = snoise( u_seed + vec2(uv.x * SCALEX + dirX, uv.y * SCALEY + dirY ) * 3.);
  float yPos = mix(noise1, noise2, 0.2);

  newPos.y = yPos * u_depth; //
  //newPos.y = sinNoise(u_time) * u_depth; //debug
  v_fragPos = vec3(model * newPos); //vec3(model * vec4(newPos, 1.0));
  v_texCoord = texcoord;
  gl_Position = projection * view * model * newPos;
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
uniform sampler2D u_noise;
uniform float u_depth;
uniform vec2 u_mouse;
varying vec3 v_normal;
varying vec3 v_fragPos;
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
  float d = (n-cutoff)/(1.-cutoff);
  d = floor(d*steps)/steps;
  return d;
}

float mapRange(float val, float low1, float high1, float low2, float high2){
  return low2 + (val - low1) * (high2 - low2) / (high1 - low1);
}

void main(){
  vec2 uv = gl_FragCoord.xy / u_resolution;
  float yPos = v_fragPos.y / u_depth; //y position in 0..1 range
  vec3 lightColor = u_lightColor;
  vec3 ambientLightColor = u_ambientLightColor;

  // Slice hills into layers
  const int steps = 3; //number of layers
  const int offset = 0; //start counting n layers with an offset

  // Set pixel color
  float h = 0.0; //0.68 + getDepth(yPos, 0.0, float(steps)) * 0.28;
  float s = 0.0;
  float v = 0.0;

  float d = getDepth(yPos, 0.0, float(steps));
  const float thickness = 0.08; //for making simple gradients

  for (int i = offset; i<steps; i++){
    float val = float(float(i))/float(steps);

    if (yPos <= val && yPos > val - thickness){
      v += mapRange(yPos, val - thickness, val, 0.0, 0.7); // gradient falloff
      //h = d * 0.8 + 0.1;
      //s = 0.3;
    }

    // //Inverse
    // if (yPos <= val + thickness && yPos > val){
    //   v = mapRange(yPos, val + thickness, val, 0.0, 0.5);
    // }

    // // Thin solid line
    if (yPos <= val && yPos > val - 0.003){
      v = 1.;
    }
  }

// Pre-lighting noise
  // Add noise and clamp values
  v *= floor((whiteNoise(uv) * v) + 0.85);
  v += floor( (whiteNoise(uv * 3.) + 0.4) * floor(v + 0.9) );
  v = clamp(v, 0.0, 1.0);

  // Add lighting
  vec3 ambient = u_ambientIntensity * ambientLightColor;
  vec3 lightPos = u_lightPos;

  vec3 normalized = normalize(v_normal);
  vec3 lightDirection = normalize(lightPos - v_fragPos);
  vec3 diffuse = max(dot(normalized, lightDirection), 0.0) * lightColor;

  vec3 position = vec3(-u_mouse.x * u_resolution.x/5., -500.0, u_mouse.y * u_resolution.y/5.); //set light origin
  vec3 viewDirection = normalize(position - v_fragPos);
  vec3 reflectionDirection = reflect(-lightDirection, normalized);

  float spec = pow(max(dot(viewDirection, reflectionDirection), 0.0), 4.);
  vec3 specular = u_specularIntensity * spec * lightColor;

  vec3 lighting = diffuse + ambient + specular;

  // Set diffuse color
  vec3 diffuseColor = hsv2rgb(vec3(h, s, v));
  vec3 col = diffuseColor * lighting; //add lighting
  //vec3 col = diffuseColor;

// Post-lighting noise
  // Add noise and clamp values
  //col = floor((whiteNoise(uv) * v) + col + 0.4); //mono noise
  //col *= floor((whiteNoise(uv) * v) + 0.85); //grey noise
  //col +=  (whiteNoise(uv * 3.) + 0.4) * floor(col + 0.8) * 0.2; //second layer
  //col = clamp(col, 0.0, 1.0);

  // Add vertical gradient
  col *= 0.35 + uv.y * 0.65;

  vec4 finalColor = vec4(col, 1.0);
  //finalColor += vec4(1.0, (1.-yPos), (1.-yPos) * 0.4, 1.0); //debug override
  gl_FragColor = finalColor;
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


// Initialize WebGL
let mousePos = {x: 0, y: 0.5};

function main(){
  const m4 = twgl.m4;
  const gl = document.getElementById("c").getContext("webgl", {alpha: false});
  const shader = twgl.createProgramInfo(gl, [vs, fs]);

  const vertices = 256;
  const offset = 0; //(vertices - 1) / 2;
  const stretchX = 2.2;
  const stretchY = 2.6;
  const strips = vertices - 1;
  const strip_length = vertices * 2;

  let position = [];
  let normal = [];
  let indices = [];

  const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
  for(let x = 0; x < vertices; x++) {
    for(let z = 0; z < vertices; z++) {
      //arrays.position.push(x - offset, ctx.getImageData(x, z, 1, 1).data[0] / z_compression, z - offset);
      position.push((x * aspect * stretchX) - (vertices * stretchX * aspect)/2, 0, (z * stretchY) - (vertices * stretchY)/2);
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
    position: { numComponents: 3, data: position },
    normal: { numComponents: 3, data: normal },
    indices: { numComponents: 1, data: indices }
  };

  const buffers = twgl.createBufferInfoFromArrays(gl, arrays);

  const textures = twgl.createTextures(gl, {
    noise: { src: "js/noise.png", mag: gl.LINEAR }
  });

  const uniforms = {
    model: [],
    view: [],
    projection: [],
    u_lightPos: [0, 200, 0], //origin of spotlight
    u_lightColor: [1, 1, 1], //color of spotlight
    u_ambientLightColor: [1, 1, 1], //ambient light color
    u_ambientIntensity: 0.4, //ambient light intensity
    u_specularIntensity: 0.5, //specular intensity of spotlight
    u_time: 0, //time
    u_resolution: [gl.canvas.width, gl.canvas.height], //canvas resolution
    u_direction: [0.0, 1.0], //direction of movement
    u_noise: textures.noise, //static noise texture
    u_mouse: [mousePos.x, mousePos.y], //mouse position, could also use for IMU on mobile
    u_depth: 70, //how tall are the mountains
    u_scale: 1., //scale of the simplex noise
    u_seed: Math.random() //noise seed
  };

  gl.clearColor(0, 0, 0, 1);

  function render(time) {
    twgl.resizeCanvasToDisplaySize(gl.canvas, window.devicePixelRatio);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.frontFace(gl.CW);
    //gl.enable(gl.DEPTH_TEST);
    //gl.enable(gl.CULL_FACE);
    //gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    //gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);
    //gl.enable(gl.BLEND);
    //gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const zoom = 25; //camera zoom
    const fov = zoom * Math.PI / 130;
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const zNear = 1;
    const zFar = 10000;
    const projection = m4.perspective(fov, aspect, zNear, zFar);

    const eye = [0, 500, -500]; //camera position
    const target = [0, -125, 0];   //camera target
    const up = [0, 1, 0];       //up direction
    const camera = m4.lookAt(eye, target, up);

    uniforms.view = m4.inverse(camera);
    uniforms.model = m4.identity();
    uniforms.projection = projection;
    uniforms.u_time = time / 20000;
    uniforms.u_mouse = [mousePos.x, mousePos.y];
    uniforms.u_resolution = [2500,2500], //[gl.canvas.width, gl.canvas.height];
    //uniforms.u_direction = [mousePos.x, mousePos.y];
    //uniforms.u_depth = Math.sin(time/200) * 100 //wobbly mountains

    gl.useProgram(shader.program);

    twgl.setBuffersAndAttributes(gl, shader, buffers);
    twgl.setUniforms(shader, uniforms);
    twgl.drawBufferInfo(gl, gl.TRIANGLE_STRIP, buffers);

    requestAnimationFrame(render);
  }

  window.addEventListener('mousemove', e => {
    const pos = getNoPaddingNoBorderCanvasRelativeMousePosition(e, gl.canvas);
    const x = pos.x / gl.canvas.width  *  2 - 1;
    const y = pos.y / gl.canvas.height * -2 + 1;

    mousePos.x = x;
    mousePos.y = y;
  });


  requestAnimationFrame(render);
}

main();

