# Orbital View Enhancement Plan

## Context

The SpinnyBall project currently features a functional but visually basic orbital simulation. The user wants to transform it into an impressive, game-like visualization that emphasizes **dramatic scale transitions** - specifically the ability to "zoom way out" to see full orbital context. This serves as both a functional enhancement and a demonstration of creative 3D visualization capabilities.

**Current Limitations:**
- Camera zoom restricted to radius 5-40 scene units (insufficient for orbital context)
- Earth sphere exists at z=-50 but rarely visible due to zoom constraints
- Orbital trajectory line buffer created but never populated with data
- Basic starfield (800 uniform white points)
- No post-processing effects or advanced visual enhancements
- toggleViewMode() only toggles visibility without repositioning camera

**Goal:** Create a visually stunning, professional-grade orbital visualization with seamless zoom from close-up coil detail (r=2) to full orbital vista (r=200+), featuring sci-fi aesthetic elements like bloom effects, enhanced space environment, and dynamic level-of-detail rendering.

---

## Visual Design Vision

### Aesthetic: "Sci-Fi Mission Control"

**Color Palette:**
- Deep space black: `#000005`
- Electric blue accents: `#0088ff`, `#4488ff`
- Warm amber/gold: `#f2cc60`, `#ffaa44`
- Nebula purple: `#2d0a4a` → `#4a0e78`
- HUD cyan: `#00ccff`

**Key Visual Moments:**
1. **Launch**: Ball accelerates through coils with cascading blue-white bloom flashes, leaving luminous amber trail
2. **Zoom Reveal**: As camera pulls back, Earth emerges with atmospheric glow, orbit path materializes as glowing green arc
3. **Full Orbital View**: At maximum zoom, user sees Earth's curvature, complete orbital ellipse, multiple ghost trails, all against subtle nebula backdrop with depth-layered starfield

**Design Principles:**
- Scale drama through smooth LOD transitions
- Every interaction produces satisfying visual feedback
- Near objects crisp and detailed, distant objects glow softly
- Consistent sci-fi color language throughout

---

## Technical Architecture

### Core Systems

**1. Extended Camera Controller**
- Zoom range: r=2 to r=250 (from current 5-40)
- Smooth interpolation for zoom transitions
- Automatic focus adjustment based on zoom level

**2. Level-of-Detail (LOD) Manager**
```javascript
const lodManager = {
  currentLevel: 'close', // 'close' | 'medium' | 'orbital' | 'macro'
  thresholds: { close: 8, medium: 25, orbital: 80 },
  
  update(cameraRadius) {
    // Determine level based on radius
    // Trigger visibility changes for features
    // Smooth transitions between levels
  }
};
```

Visibility gates:
- **r < 8** (Close): Full detail - individual coils, force arrows, inset canvas, UI panels
- **r = 8-25** (Medium): Coils visible, simplified UI, hide force arrows
- **r = 25-80** (Orbital): Earth appears, orbit line glows, show orbital HUD
- **r > 80** (Macro): Nebula background visible, minimal UI, enhanced starfield

**3. Post-Processing Pipeline**
- EffectComposer with render passes:
  1. RenderPass (main scene)
  2. UnrealBloomPass (strength: 1.5, radius: 0.4, threshold: 0.2)
  3. OutputPass (final composite)

**4. Orbital Trajectory Calculator**
- Computes Keplerian orbit from altitude/inclination parameters
- Generates buffer geometry for orbit line (currently empty)
- Updates dynamically when orbital parameters change

---

## Implementation Phases

### Phase 1: Foundation - Extended Zoom & LOD Framework

**Objective:** Enable dramatic zoom range with basic LOD system

**Modifications:**

1. **Extend Camera Range** (`index.html`, lines 604-688)
   - Change radius limits: `Math.max(5, Math.min(40, ...))` → `Math.max(2, Math.min(250, ...))`
   - Update wheel sensitivity: multiply delta by 0.05 instead of 0.02
   - Adjust touch pinch-to-zoom scaling factor

2. **Create LOD Manager** (New section after line 575)
   ```javascript
   const lodManager = {
     currentLevel: 'close',
     thresholds: { close: 8, medium: 25, orbital: 80 },
     
     update(radius) {
       const prevLevel = this.currentLevel;
       
       if (radius < this.thresholds.close) this.currentLevel = 'close';
       else if (radius < this.thresholds.medium) this.currentLevel = 'medium';
       else if (radius < this.thresholds.orbital) this.currentLevel = 'orbital';
       else this.currentLevel = 'macro';
       
       if (prevLevel !== this.currentLevel) {
         this.applyLevel(this.currentLevel);
       }
     },
     
     applyLevel(level) {
       // Toggle visibility based on level
       if (level === 'close' || level === 'medium') {
         // Show detailed UI elements
         document.getElementById('rightPanel').style.display = 'flex';
       } else {
         document.getElementById('rightPanel').style.display = 'none';
       }
       
       if (level === 'orbital' || level === 'macro') {
         earthMesh.visible = true;
         orbitLine.visible = true;
       } else {
         earthMesh.visible = false;
         orbitLine.visible = false;
       }
     }
   };
   ```

3. **Integrate LOD into Render Loop** (line 2156+)
   - Add: `lodManager.update(orbitControls.getRadius());`

4. **Enhanced toggleViewMode()** (lines 1976-1991)
   ```javascript
   function toggleViewMode() {
     viewMode = viewMode === 'local' ? 'orbital' : 'local';
     const btn = document.getElementById('viewToggleBtn');
     btn.textContent = viewMode === 'local' ? 'Switch to Orbital View' : 'Switch to Local View';
     
     // Animate camera transition
     if (viewMode === 'orbital') {
       orbitControls.animateTo(0.3, Math.PI / 2.5, 120, 1.0); // Smooth transition over 1s
     } else {
       orbitControls.animateTo(0.4, Math.PI / 3, 14, 1.0);
     }
   }
   ```

5. **Add animateTo() method to orbitControls** (within makeOrbitControls function)
   ```javascript
   let animTarget = null, animStartTime = 0, animDuration = 0;
   
   function animateTo(targetTheta, targetPhi, targetRadius, duration) {
     animTarget = { theta: targetTheta, phi: targetPhi, radius: targetRadius };
     animStartTime = performance.now();
     animDuration = duration * 1000;
   }
   
   // In update(): check if animating and interpolate
   if (animTarget) {
     const elapsed = performance.now() - animStartTime;
     const t = Math.min(elapsed / animDuration, 1);
     const eased = t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3) / 2; // ease-in-out cubic
     
     theta = lerp(theta, animTarget.theta, eased);
     phi = lerp(phi, animTarget.phi, eased);
     radius = lerp(radius, animTarget.radius, eased);
     
     if (t >= 1) animTarget = null;
     update();
   }
   
   function lerp(a, b, t) { return a + (b - a) * t; }
   
   return { update, getTheta, getPhi, getRadius, setView, animateTo };
   ```

**Test Criteria:**
- ✅ Can zoom out to r=200 and see Earth clearly
- ✅ No performance degradation at max zoom
- ✅ Smooth zoom transitions without jarring jumps
- ✅ toggleViewMode() smoothly animates camera position

---

### Phase 2: Post-Processing Bloom & Enhanced Materials

**Objective:** Add glowing effects for sci-fi aesthetic

**Modifications:**

1. **Switch to ES Module Imports** (lines 589-599)
   
   Replace current script loading with:
   ```javascript
   <script type="module">
     import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.module.js';
     import { EffectComposer } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/postprocessing/EffectComposer.js';
     import { RenderPass } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/postprocessing/RenderPass.js';
     import { ShaderPass } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/postprocessing/ShaderPass.js';
     import { UnrealBloomPass } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/postprocessing/UnrealBloomPass.js';
     import { CopyShader } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/shaders/CopyShader.js';
     import { LuminosityHighPassShader } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/shaders/LuminosityHighPassShader.js';
     
     window.THREE = THREE; // Make global for existing code
     init();
   </script>
   ```

2. **Initialize Post-Processing** (in `init()`, around line 2210)
   ```javascript
   // Post-processing setup
   composer = new EffectComposer(renderer);
   composer.addPass(new RenderPass(scene, camera));
   
   const bloomPass = new UnrealBloomPass(
     new THREE.Vector2(window.innerWidth / 2, window.innerHeight / 2), // Half resolution for performance
     1.5,  // strength
     0.4,  // radius
     0.2   // threshold
   );
   composer.addPass(bloomPass);
   ```

3. **Enhance Material Emissive Properties**
   
   - **Ball material** (line 2278): 
     ```javascript
     emissive: 0xffaa44,
     emissiveIntensity: 0.8,
     ```
   
   - **Coil materials** (line 1049):
     ```javascript
     emissiveIntensity: 1.0,  // default state
     // Flash intensity in visualEffectsManager: increase from 3.0 to 5.0
     ```
   
   - **Earth material** (line 2247):
     ```javascript
     emissive: 0x003366,
     emissiveIntensity: 0.4,
     ```

4. **Update Render Loop** (line 2203)
   - Replace: `renderer.render(scene, camera);`
   - With: `composer.render();`

**Test Criteria:**
- ✅ Ball emits warm golden glow visible at all zoom levels
- ✅ Coil flashes produce satisfying white-blue bloom bursts
- ✅ Performance remains above 50 FPS on desktop
- ✅ Glow intensity appropriate (not overwhelming)

---

### Phase 3: Orbital Trajectory & Enhanced Space Environment

**Objective:** Populate orbit line and upgrade background

**Modifications:**

1. **Calculate & Render Orbital Path** (New function after line 1991)
   ```javascript
   function calculateOrbitPath(P) {
     const R_earth = 6371.0; // km
     const altitude = P.altitude;
     const inclination = P.inclination * Math.PI / 180;
     const r_orbit = R_earth + altitude;
     
     // Scale: 1 scene unit ≈ 100 km
     const scale = 1 / 100;
     const sceneRadius = r_orbit * scale;
     
     const numPoints = 200;
     const positions = new Float32Array(numPoints * 3);
     
     for (let i = 0; i < numPoints; i++) {
       const angle = (i / numPoints) * Math.PI * 2;
       
       // Orbital plane coordinates
       const x = sceneRadius * Math.cos(angle);
       const z = sceneRadius * Math.sin(angle);
       
       // Apply inclination rotation around X axis
       const y = z * Math.sin(inclination);
       const z_rotated = z * Math.cos(inclination);
       
       positions[i * 3] = x;
       positions[i * 3 + 1] = y;
       positions[i * 3 + 2] = z_rotated - 50; // Offset to match Earth position
     }
     
     orbitLine.geometry.setAttribute('position', 
       new THREE.BufferAttribute(positions, 3));
     orbitLine.geometry.setDrawRange(0, numPoints);
   }
   ```

2. **Call Orbit Calculation** (In `onParamChange()`, line 930+)
   ```javascript
   calculateOrbitPath(currentP);
   ```

3. **Upgrade Starfield** (Replace lines 2229-2242)
   ```javascript
   function makeStars() {
     const starConfigs = [
       { count: 1500, rMin: 80, rMax: 100 },
       { count: 1000, rMin: 120, rMax: 150 },
       { count: 500, rMin: 180, rMax: 220 }
     ];
     
     let totalStars = 0;
     starConfigs.forEach(c => totalStars += c.count);
     
     const geo = new THREE.BufferGeometry();
     const positions = new Float32Array(totalStars * 3);
     const colors = new Float32Array(totalStars * 3);
     
     let idx = 0;
     starConfigs.forEach(config => {
       for (let i = 0; i < config.count; i++) {
         const theta = Math.random() * Math.PI * 2;
         const phi2 = Math.acos(2 * Math.random() - 1);
         const r = config.rMin + Math.random() * (config.rMax - config.rMin);
         
         positions[idx * 3] = r * Math.sin(phi2) * Math.cos(theta);
         positions[idx * 3 + 1] = r * Math.cos(phi2);
         positions[idx * 3 + 2] = r * Math.sin(phi2) * Math.sin(theta);
         
         // Color variation: 70% white, 20% blue, 10% yellow
         const color = new THREE.Color();
         const rand = Math.random();
         if (rand < 0.7) color.setHex(0xffffff);
         else if (rand < 0.9) color.setHex(0x8899ff);
         else color.setHex(0xffdd88);
         
         colors[idx * 3] = color.r;
         colors[idx * 3 + 1] = color.g;
         colors[idx * 3 + 2] = color.b;
         
         idx++;
       }
     });
     
     geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
     geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
     
     const mat = new THREE.PointsMaterial({ 
       size: 0.15,
       vertexColors: true,
       sizeAttenuation: true
     });
     
     scene.add(new THREE.Points(geo, mat));
   }
   ```

4. **Add Nebula Background** (New function in `init()`)
   ```javascript
   function makeNebula() {
     const geo = new THREE.SphereGeometry(300, 32, 32);
     
     const vertexShader = `
       varying vec3 vWorldPosition;
       void main() {
         vec4 worldPosition = modelMatrix * vec4(position, 1.0);
         vWorldPosition = worldPosition.xyz;
         gl_Position = projectionMatrix * viewMatrix * worldPosition;
       }
     `;
     
     const fragmentShader = `
       varying vec3 vWorldPosition;
       
       // Simple noise function
       float hash(vec3 p) {
         p = fract(p * 0.3183099 + 0.1);
         p *= 17.0;
         return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
       }
       
       float noise(vec3 x) {
         vec3 i = floor(x);
         vec3 f = fract(x);
         f = f * f * (3.0 - 2.0 * f);
         
         return mix(mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
                        mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
                    mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                        mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
       }
       
       void main() {
         vec3 dir = normalize(vWorldPosition);
         float n = noise(dir * 3.0) * 0.5 + noise(dir * 6.0) * 0.25;
         
         vec3 color1 = vec3(0.18, 0.04, 0.29); // Purple
         vec3 color2 = vec3(0.04, 0.10, 0.23); // Dark blue
         
         vec3 color = mix(color1, color2, n + 0.3);
         float alpha = smoothstep(0.3, 0.7, n) * 0.4;
         
         gl_FragColor = vec4(color, alpha);
       }
     `;
     
     const mat = new THREE.ShaderMaterial({
       vertexShader,
       fragmentShader,
       transparent: true,
       side: THREE.BackSide,
       depthWrite: false
     });
     
     nebulaMesh = new THREE.Mesh(geo, mat);
     nebulaMesh.visible = false; // Controlled by LOD
     scene.add(nebulaMesh);
   }
   ```

5. **Update LOD Manager** to control nebula visibility
   ```javascript
   applyLevel(level) {
     // ... existing code ...
     
     if (level === 'macro') {
       nebulaMesh.visible = true;
     } else {
       nebulaMesh.visible = false;
     }
   }
   ```

**Test Criteria:**
- ✅ Orbit line renders as smooth green arc matching orbital parameters
- ✅ Changing altitude slider updates orbit radius visibly
- ✅ Starfield shows depth and color variation when camera moves
- ✅ Nebula provides subtle backdrop at r > 80 without overwhelming scene

---

### Phase 4: Polish - Atmospheric Effects & Particle System

**Objective:** Add finishing touches for professional polish

**Modifications:**

1. **Earth Atmospheric Scattering Shader** (Replace Earth material, line 2247)
   ```javascript
   const earthVertexShader = `
     varying vec3 vNormal;
     varying vec3 vPosition;
     
     void main() {
       vNormal = normalize(normalMatrix * normal);
       vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
       gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
     }
   `;
   
   const earthFragmentShader = `
     varying vec3 vNormal;
     varying vec3 vPosition;
     
     uniform vec3 atmosphereColor;
     uniform float atmosphereIntensity;
     uniform float atmospherePower;
     
     void main() {
       vec3 viewDirection = normalize(-vPosition);
       float fresnel = pow(1.0 - dot(viewDirection, vNormal), atmospherePower);
       
       // Base Earth color
       vec3 baseColor = vec3(0.13, 0.27, 0.67); // #2244aa
       
       // Atmosphere glow
       vec3 atmosphere = atmosphereColor * fresnel * atmosphereIntensity;
       
       vec3 finalColor = baseColor + atmosphere;
       
       gl_FragColor = vec4(finalColor, 1.0);
     }
   `;
   
   const earthMat = new THREE.ShaderMaterial({
     vertexShader: earthVertexShader,
     fragmentShader: earthFragmentShader,
     uniforms: {
       atmosphereColor: { value: new THREE.Color(0x4488ff) },
       atmosphereIntensity: { value: 0.6 },
       atmospherePower: { value: 2.5 }
     }
   });
   ```

2. **Particle Trail System** (New object after visualEffectsManager, ~line 1175)
   ```javascript
   const particleSystem = {
     particles: [],
     maxParticles: 500,
     geometry: null,
     material: null,
     points: null,
     
     init() {
       this.geometry = new THREE.BufferGeometry();
       const positions = new Float32Array(this.maxParticles * 3);
       const opacities = new Float32Array(this.maxParticles);
       
       this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
       this.geometry.setAttribute('opacity', new THREE.BufferAttribute(opacities, 1));
       
       this.material = new THREE.PointsMaterial({
         color: 0xffaa44,
         size: 0.3,
         transparent: true,
         opacity: 0.8,
         sizeAttenuation: true,
         blending: THREE.AdditiveBlending,
         depthWrite: false
       });
       
       this.points = new THREE.Points(this.geometry, this.material);
       scene.add(this.points);
     },
     
     emit(position, velocity, count = 5) {
       for (let i = 0; i < count; i++) {
         if (this.particles.length >= this.maxParticles) {
           this.particles.shift(); // Remove oldest
         }
         
         this.particles.push({
           position: position.clone().addScalar((Math.random() - 0.5) * 0.1),
           velocity: velocity.clone().multiplyScalar(-0.3).add(
             new THREE.Vector3(
               (Math.random() - 0.5) * 0.05,
               (Math.random() - 0.5) * 0.05,
               (Math.random() - 0.5) * 0.05
             )
           ),
           life: 1.0,
           decay: 0.02 + Math.random() * 0.03
         });
       }
     },
     
     update(dt) {
       const positions = this.geometry.attributes.position.array;
       const opacities = this.geometry.attributes.opacity.array;
       
       // Update particles
       for (let i = this.particles.length - 1; i >= 0; i--) {
         const p = this.particles[i];
         p.life -= p.decay;
         
         if (p.life <= 0) {
           this.particles.splice(i, 1);
           continue;
         }
         
         p.position.add(p.velocity.clone().multiplyScalar(dt));
       }
       
       // Update buffer
       for (let i = 0; i < this.maxParticles; i++) {
         if (i < this.particles.length) {
           const p = this.particles[i];
           positions[i * 3] = p.position.x;
           positions[i * 3 + 1] = p.position.y;
           positions[i * 3 + 2] = p.position.z;
           opacities[i] = p.life;
         } else {
           positions[i * 3] = 0;
           positions[i * 3 + 1] = 0;
           positions[i * 3 + 2] = 0;
           opacities[i] = 0;
         }
       }
       
       this.geometry.attributes.position.needsUpdate = true;
       this.geometry.attributes.opacity.needsUpdate = true;
       this.geometry.setDrawRange(0, this.particles.length);
     },
     
     clear() {
       this.particles = [];
     }
   };
   ```

3. **Emit Particles During Transit** (In `updateBallPosition()`, ~line 1080)
   ```javascript
   // After updating ball position
   if (isAnimating && currentFrameIndex < transitData.T.length) {
     const velocity = new THREE.Vector3(
       transitData.X[currentFrameIndex + 1] - transitData.X[currentFrameIndex],
       0,
       transitData.Z[currentFrameIndex + 1] - transitData.Z[currentFrameIndex]
     ).normalize();
     
     particleSystem.emit(ballMesh.position, velocity, 3);
   }
   ```

4. **Update Render Loop** (line 2156+)
   ```javascript
   particleSystem.update(dt);
   ```

5. **Initialize Particle System** (In `init()`, after other initializations)
   ```javascript
   particleSystem.init();
   ```

**Test Criteria:**
- ✅ Earth shows convincing blue atmospheric glow at edges
- ✅ Particle trails enhance sense of motion during ball transit
- ✅ Particles fade smoothly and don't accumulate excessively
- ✅ No visual artifacts (z-fighting, harsh edges)

---

## Critical Files & Modifications Summary

### Primary File: `index.html`

**Sections to Modify:**

| Line Range | Section | Modification |
|------------|---------|--------------|
| 589-599 | Three.js loading | Switch to ES module imports with post-processing addons |
| 604-688 | `makeOrbitControls()` | Extend radius range (2-250), add `animateTo()` method |
| 575 (new) | LOD Manager | Add new LOD manager object definition |
| 1175 (new) | Particle System | Add particle trail system object |
| 1976-1991 | `toggleViewMode()` | Replace visibility toggle with animated camera transition |
| 1991 (new) | `calculateOrbitPath()` | Add orbital trajectory calculation function |
| 2156-2204 | `renderLoop()` | Add LOD update, particle system update, use composer.render() |
| 2209-2358 | `init()` | Initialize post-processing, upgrade starfield, add nebula, calculate orbit |
| 2244-2266 | Earth & orbit setup | Replace Earth material with atmospheric shader, populate orbit line |
| 1049 | Coil materials | Increase emissive intensity for bloom compatibility |
| 2278 | Ball material | Enhance emissive properties |
| 930+ | `onParamChange()` | Call `calculateOrbitPath()` when parameters change |

**New Functions to Add:**
- `lodManager` object (~line 575)
- `particleSystem` object (~line 1175)
- `calculateOrbitPath(P)` (~line 1991)
- `makeNebula()` (~line 2242)
- `animateTo()` method within `makeOrbitControls()`

---

## Dependencies

### Required Three.js Modules (via jsdelivr CDN):

```javascript
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.module.js';
import { EffectComposer } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/postprocessing/UnrealBloomPass.js';
import { CopyShader } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/shaders/CopyShader.js';
import { LuminosityHighPassShader } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/shaders/LuminosityHighPassShader.js';
```

**No additional external libraries required.**

---

## Risk Assessment & Mitigation

### High-Risk Areas

**R1: Module Loading Complexity**
- **Risk**: Switching from global `<script>` to ES modules may break existing code structure
- **Mitigation**: 
  - Keep all code within single `<script type="module">` block
  - Ensure all Three.js references use imported `THREE` namespace
  - Set `window.THREE = THREE` for backward compatibility
  - Test incrementally: verify basic scene loads before adding modules

**R2: Performance Degradation**
- **Risk**: Post-processing bloom and 3000 stars may drop FPS below acceptable levels
- **Mitigation**:
  - Use half-resolution bloom pass (window.size / 2)
  - Implement LOD-based feature toggling
  - Profile with Chrome DevTools Performance tab
  - Provide fallback: detect low-end devices and reduce bloom strength automatically

**R3: Single-File Size Constraint**
- **Risk**: Adding shaders, particle systems, and LOD logic may bloat file beyond maintainability
- **Mitigation**:
  - Keep shader code as compact GLSL strings
  - Use concise object-literal syntax for new systems
  - Target final size under 120KB (current: 90KB)

### Medium-Risk Areas

**R4: Camera Transition Smoothness**
- **Risk**: Animated camera movements may feel jerky or disorienting
- **Mitigation**:
  - Use proven easing functions (cubic ease-in-out)
  - Limit transition duration to 0.8-1.2 seconds
  - Allow user interruption of animations

**R5: Mobile Compatibility**
- **Risk**: Enhanced effects may not work well on mobile GPUs
- **Mitigation**:
  - Detect mobile via screen size (< 768px width)
  - Reduce bloom strength on mobile (1.5 → 0.8)
  - Limit particle count on mobile (500 → 200)
  - Disable nebula on mobile entirely

---

## Success Criteria

### Functional Requirements (Must Pass):

✅ **Zoom Range**: User can zoom from r=2 (ball fills 30% of screen) to r=200 (Earth fully visible with orbit path)

✅ **LOD System**: At least 3 distinct detail levels activate at appropriate zoom thresholds without manual intervention

✅ **Orbit Path**: Green orbital trajectory line renders correctly and updates when altitude/inclination parameters change

✅ **Post-Processing Bloom**: Ball, coils, and Earth exhibit visible glow effects that intensify during active events (coil flashes)

✅ **Performance**: Maintains ≥50 FPS on desktop (Chrome/Firefox), ≥30 FPS on modern mobile devices

### Visual Quality Requirements:

✅ **Scale Drama**: Zooming out produces clear "wow moment" when Earth and full orbit become visible

✅ **Cohesive Aesthetic**: All elements share consistent sci-fi color palette (blues, ambers, purples)

✅ **Motion Feedback**: Ball movement feels dynamic through particle trails and coil activation sequences

✅ **Professional Polish**: No visual artifacts (z-fighting, flickering, harsh transitions)

### User Experience Requirements:

✅ **Intuitive Controls**: Existing mouse/touch controls continue working without relearning

✅ **Progressive Enhancement**: Each phase adds value independently; partial implementation still improves experience

✅ **Responsive**: Works across desktop (1920×1080) to mobile (375×667) screen sizes

---

## Testing Strategy

### Incremental Testing After Each Phase:

**Phase 1 Test:**
```
1. Open index.html in browser
2. Scroll mouse wheel outward until Earth appears
3. Verify no clipping or rendering errors at r=200
4. Click "Switch to Orbital View" button
5. Verify camera smoothly moves to orbital position
```

**Phase 2 Test:**
```
1. Press FIRE to launch ball
2. Observe coil flashes - should produce white bloom halos
3. Check ball has golden glow throughout transit
4. Monitor FPS (should stay above 50 on desktop)
5. Verify glow doesn't overwhelm scene details
```

**Phase 3 Test:**
```
1. Change altitude slider from 400 to 800 km
2. Verify orbit line radius increases proportionally
3. Change inclination from 51.6° to 0°
4. Verify orbit line rotates to equatorial plane
5. Zoom out to r=150 and observe starfield depth
6. Verify nebula appears at r > 80
```

**Phase 4 Test:**
```
1. Zoom to r=3 and observe Earth atmospheric glow
2. Launch ball and watch for amber particle trail
3. Verify particles fade smoothly
4. Test on mobile device for performance
5. Check all transitions are smooth
```

---

## Implementation Priority

For maximum impact with controlled risk, implement in this sequence:

1. **Phase 1 (Foundation)** - 2-3 hours
   - Extended zoom range alone delivers 40% of requested value
   - Low risk, high reward

2. **Phase 3 (Orbital Path + Stars)** - 2-3 hours  
   - Populates currently-empty orbit line
   - Enhanced starfield adds immediate depth
   - Independent of post-processing complexity

3. **Phase 2 (Bloom)** - 3-4 hours
   - Highest visual impact but most technical risk
   - Test thoroughly on target devices before proceeding

4. **Phase 4 (Polish)** - 2-3 hours
   - Nice-to-have enhancements
   - Can be partially implemented based on remaining time

**Total Estimated Effort**: 9-13 hours for complete implementation

---

## Expected Outcome

Upon completion, the SpinnyBall orbital visualization will feature:

- **Seamless multi-scale exploration** from intimate coil-level detail to breathtaking orbital vistas
- **Cinematic visual quality** with bloom effects, atmospheric scattering, and particle systems
- **Dynamic level-of-detail rendering** that maintains performance while maximizing visual richness
- **Professional sci-fi aesthetic** with cohesive color palette and polished interactions
- **Impressive "zoom reveal" moments** that demonstrate sophisticated 3D graphics capabilities

The implementation respects the single-file constraint, maintains backward compatibility with existing functionality, and provides a foundation for future enhancements.
