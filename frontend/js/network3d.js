/**
 * network3d.js — Three.js 3D Live Network Visualization
 * Real-time 3D scene: base station towers, UE particles,
 * animated resource allocation beams (eMBB/URLLC/mMTC).
 * All data driven from live WebSocket pipeline.
 */

window.Network3D = (() => {
  let renderer, scene, camera, controls;
  let animId;
  let bsMeshes  = {};   // bs_id → {tower, ring, label}
  let ueMeshes  = {};   // "bs_ue" → mesh
  let connectionLines = {}; // "bs_ue" → {line, curve, lastPRB}
  let ripples = [];     // Array of {mesh, bs_id, life}
  let dataBeams = [];   // Active data packet meshes
  let clock;
  let initialized = false;

  const SCALE  = 0.001;  // metres → scene units
  const COLORS = {
    eMBB:  0x0088ff,
    URLLC: 0xff4466,
    mMTC:  0x00ff88,
    BS:    0xffb300,
    UE:    0x44aaff,
    BG:    0x010810,
  };

  // ── init ──────────────────────────────────────────────────────── //
  function init() {
    const container = document.getElementById('scene-container');
    if (!container || initialized) return;

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x000000, 0.4); // Semi-transparent black to drastically darken CSS gradient behind it
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x02050a, 0.0006); // Darker fog for more neon contrast

    // Camera
    camera = new THREE.PerspectiveCamera(
      55, container.clientWidth / container.clientHeight, 0.1, 5000
    );
    camera.position.set(0, 350, 500);
    camera.lookAt(0, 0, 0);

    // Orbit controls (loaded via CDN)
    if (THREE.OrbitControls) {
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.minDistance   = 200;
      controls.maxDistance   = 2000;
      controls.maxPolarAngle = Math.PI / 2.1;
    }

    // Lighting
    scene.add(new THREE.AmbientLight(0x0a1428, 2));
    const dirLight = new THREE.DirectionalLight(0x00aaff, 1.5);
    dirLight.position.set(200, 400, 200);
    scene.add(dirLight);

    // Ground plane (hex grid feel)
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(3000, 3000, 60, 60),
      new THREE.MeshStandardMaterial({
        color: 0x011a44, wireframe: false,
        roughness: 0.95, metalness: 0.2,
        transparent: true, opacity: 0.15, // Let bg-canvas show through
      })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    // Grid overlay
    const grid = new THREE.GridHelper(3000, 60, 0x001a33, 0x001a33);
    grid.position.y = 0.1;
    scene.add(grid);

    clock = new THREE.Clock();

    // Resize listener
    window.addEventListener('resize', onResize);

    // Animate
    animate();
    initialized = true;

    // Build initial BS towers from API
    fetch('http://localhost:8000/api/network/positions')
      .then(r => r.ok ? r.json() : null)
      .then(d => { if(d) buildBSTowers(d.bs || []); })
      .catch(() => {});
  }

  // ── Star field ────────────────────────────────────────────────── //
  function buildStarField() {
    const geo = new THREE.BufferGeometry();
    const n = 3000;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i*3]   = (Math.random()-0.5)*4000;
      pos[i*3+1] = Math.random()*1500 + 100;
      pos[i*3+2] = (Math.random()-0.5)*4000;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
      color: 0x224488, size: 1.5, transparent: true, opacity: 0.6
    });
    scene.add(new THREE.Points(geo, mat));
  }
  
  // ── Tower Labels (3D Sprites) ────────────────────────────────── //
  function createTowerLabel(text) {
    const canvas = document.createElement('canvas');
    const ctx =  canvas.getContext('2d');
    canvas.width = 256; canvas.height = 128;
    
    // Background
    ctx.fillStyle = 'rgba(5, 10, 20, 0.8)';
    ctx.roundRect(40, 40, 176, 48, 8);
    ctx.fill();
    ctx.strokeStyle = '#ffb300';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Text
    ctx.font = 'bold 32px JetBrains Mono';
    ctx.fillStyle = '#ffb300';
    ctx.textAlign = 'center';
    ctx.fillText(text, 128, 74);

    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(100, 50, 1);
    return sprite;
  }

  // ── BS Towers ─────────────────────────────────────────────────── //
  function buildBSTowers(bsList) {
    bsList.forEach(bs => {
      const x = bs.x * SCALE * 1000;
      const z = bs.y * SCALE * 1000;
      const group = new THREE.Group();

      // Tower body
      const body = new THREE.Mesh(
        new THREE.CylinderGeometry(5, 12, 120, 8),
        new THREE.MeshStandardMaterial({
          color: 0x102040, metalness: 0.8, roughness: 0.3,
          emissive: 0x002244, emissiveIntensity: 0.3
        })
      );
      body.position.y = 60;
      body.castShadow = true;
      group.add(body);

      // Top beacon
      const beacon = new THREE.Mesh(
        new THREE.SphereGeometry(9, 16, 16),
        new THREE.MeshStandardMaterial({
          color: COLORS.BS, emissive: COLORS.BS, emissiveIntensity: 1.5
        })
      );
      beacon.position.y = 125;
      group.add(beacon);

      // Glow ring
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(26, 2.5, 8, 40),
        new THREE.MeshStandardMaterial({
          color: COLORS.BS, emissive: COLORS.BS,
          emissiveIntensity: 0.8, transparent: true, opacity: 0.7
        })
      );
      ring.position.y = 75;
      ring.rotation.x = Math.PI / 2;
      group.add(ring);

      // Point light at BS
      const light = new THREE.PointLight(0xffaa00, 1.5, 300);
      light.position.y = 120;
      group.add(light);

      group.position.set(x, 0, z);
      scene.add(group);

      // Floating Label
      const label = createTowerLabel(`BS-${bs.bs_id}`);
      label.position.set(x, 155, z);
      scene.add(label);

      // Initial Ripple
      const rippleGeo = new THREE.RingGeometry(20, 22, 32);
      const rippleMat = new THREE.MeshBasicMaterial({ color: 0xffb300, transparent: true, opacity: 0.5, side: THREE.DoubleSide });
      const ripple = new THREE.Mesh(rippleGeo, rippleMat);
      ripple.rotation.x = -Math.PI/2;
      ripple.position.set(x, 0.5, z);
      scene.add(ripple);
      ripples.push({ mesh: ripple, x, z, scale: 1, opacity: 0.5 });

      bsMeshes[bs.bs_id] = { group, beacon, ring, light, label };
    });
  }

  // ── UE particles ─────────────────────────────────────────────── //
  const SLICE_COL_HEX = { 0: 0x0088ff, 1: 0xff4466, 2: 0x00ff88,
                           3: 0x0044ff, 4: 0xff0022, 5: 0x00cc77 };

  function updateUEs(uePositions) {
    const seenKeys = new Set();
    Object.entries(uePositions).forEach(([bsIdStr, ues]) => {
      ues.forEach(ue => {
        const key = `${bsIdStr}_${ue.ue_id}`;
        seenKeys.add(key);
        const x = ue.x * SCALE * 1000;
        const z = ue.y * SCALE * 1000;

        if (!ueMeshes[key]) {
          const col = SLICE_COL_HEX[ue.slice_id % 6] || COLORS.UE;
          const mesh = new THREE.Mesh(
            new THREE.SphereGeometry(6, 8, 8),
            new THREE.MeshStandardMaterial({
              color: col, emissive: col, emissiveIntensity: 0.6,
              transparent: true, opacity: 0.85
            })
          );
          mesh.position.set(x, 4, z);
          scene.add(mesh);
          ueMeshes[key] = mesh;
        } else {
          const mesh = ueMeshes[key];
          // Smooth lerp to new position
          mesh.position.x += (x - mesh.position.x) * 0.2;
          mesh.position.z += (z - mesh.position.z) * 0.2;
        }
      });
    });
    // Remove stale UEs
    Object.keys(ueMeshes).forEach(k => {
      if (!seenKeys.has(k)) {
        scene.remove(ueMeshes[k]);
        delete ueMeshes[k];
      }
    });
  }

  // ── Resource Connections (Tower to UE) ──────────────────────── //
  function updateConnections(bsPositions, prbMaps, slices, uePositions) {
    const seenKeys = new Set();

    bsPositions.forEach(bs => {
      const bx = bs.x * SCALE * 1000;
      const bz = bs.y * SCALE * 1000;
      const bsIdStr = String(bs.bs_id);
      const myMap = prbMaps[bsIdStr] || {};

      const ues = uePositions[bsIdStr] || [];
      ues.forEach(ue => {
        const nPRBs = myMap[String(ue.slice_id)] || 0;
        if (nPRBs < 1) return;

        const ux = ue.x * SCALE * 1000;
        const uz = ue.y * SCALE * 1000;
        const key = `${bsIdStr}_${ue.ue_id}`;
        seenKeys.add(key);
        
        const curve = new THREE.QuadraticBezierCurve3(
          new THREE.Vector3(bx, 125, bz),
          new THREE.Vector3((bx+ux)/2, 60 + (nPRBs * 0.5), (bz+uz)/2), 
          new THREE.Vector3(ux, 4, uz)
        );
        
        const col = SLICE_COL_HEX[ue.slice_id % 6] || COLORS.UE;

        // ── Spawn Active Data Beams ──
        // High PRBs = high chance to spawn a data meteor this frame
        if (Math.random() < Math.min(1.0, nPRBs / 30.0)) {
           const beamGeo = new THREE.CylinderGeometry(1.5 + Math.min(2.0, nPRBs/40), 1.5, 15 + Math.sqrt(nPRBs)*2, 6);
           beamGeo.rotateX(Math.PI / 2);
           const beamMat = new THREE.MeshBasicMaterial({
             color: col, transparent: true, opacity: 0.9, 
           });
           const beam = new THREE.Mesh(beamGeo, beamMat);
           scene.add(beam);
           dataBeams.push({
             mesh: beam,
             curve: curve,
             t: 0.0,
             // Speed scales with PRB allocation, giving a visceral fast/slow physics feel
             speed: 0.01 + Math.random() * 0.01 + Math.min(0.04, nPRBs/1000)
           });
        }

        // ── Update Core Connection Line ──
        if (!connectionLines[key]) {
          const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(16));
          const mat = new THREE.LineDashedMaterial({
            color: col, linewidth: 2, scale: 1, 
            dashSize: 3 + Math.sqrt(nPRBs), gapSize: 4,
            transparent: true, opacity: 0.15 + Math.min(0.3, nPRBs / 100)
          });
          
          const line = new THREE.Line(geo, mat);
          line.computeLineDistances();
          scene.add(line);
          connectionLines[key] = { line, curve, lastPRB: nPRBs };
        } else {
          const cl = connectionLines[key];
          cl.curve = curve;
          cl.lastPRB = nPRBs;
          cl.line.geometry.setFromPoints(curve.getPoints(16));
          cl.line.geometry.attributes.position.needsUpdate = true;
          cl.line.computeLineDistances();
          cl.line.material.dashSize = 3 + Math.sqrt(nPRBs);
          cl.line.material.opacity = 0.15 + Math.min(0.3, nPRBs / 100);
          cl.line.material.color.setHex(col);
        }
      });
    });

    // Remove stale lines
    Object.keys(connectionLines).forEach(k => {
      if (!seenKeys.has(k)) {
        scene.remove(connectionLines[k].line);
        connectionLines[k].line.geometry.dispose();
        connectionLines[k].line.material.dispose();
        delete connectionLines[k];
      }
    });
  }

  // ── Animation loop ────────────────────────────────────────────── //
  function animate() {
    animId = requestAnimationFrame(animate);
    const t = clock.getElapsedTime();

    // Pulse BS beacons & rings
    Object.values(bsMeshes).forEach(({beacon, ring, light}) => {
      const s = 1.0 + 0.2 * Math.sin(t * 2);
      beacon.scale.setScalar(s);
      ring.scale.setScalar(1.0 + 0.05 * Math.sin(t * 3));
      light.intensity = 1.2 + 0.8 * Math.sin(t * 2);
    });

    // Animate data flow dots down the lines
    Object.values(connectionLines).forEach((cl) => {
      if (cl.line.material.dashOffset !== undefined) {
        cl.line.material.dashOffset -= Math.max(0.5, Math.sqrt(cl.lastPRB || 0)*0.5);
      }
    });

    // Animate Data Beams (Meteors)
    for (let i = dataBeams.length - 1; i >= 0; i--) {
       let beam = dataBeams[i];
       beam.t += beam.speed;
       if (beam.t >= 1) {
          scene.remove(beam.mesh);
          beam.mesh.geometry.dispose();
          beam.mesh.material.dispose();
          dataBeams.splice(i, 1);
       } else {
          // Tangent alignment
          const pt1 = beam.curve.getPointAt(beam.t);
          const pt2 = beam.curve.getPointAt(Math.min(1.0, beam.t + 0.01));
          beam.mesh.position.copy(pt1);
          beam.mesh.lookAt(pt2);
          // Fade out as it hits the UE
          beam.mesh.material.opacity = Math.max(0, 1.0 - beam.t*1.1); 
       }
    }

    // UE gentle bob
    Object.values(ueMeshes).forEach((m, i) => {
      m.position.y = 4 + Math.sin(t * 1.5 + i * 0.5) * 2;
    });

    // Update Radar Ripples
    ripples.forEach((r, i) => {
      r.scale += 0.015;
      r.opacity -= 0.006;
      r.mesh.scale.setScalar(r.scale);
      r.mesh.material.opacity = r.opacity;
      
      if (r.opacity <= 0) {
        r.scale = 1;
        r.opacity = 0.5;
        r.mesh.scale.setScalar(1);
      }
    });

    controls && controls.update();
    renderer.render(scene, camera);
  }

  // ── Public API ────────────────────────────────────────────────── //
  function update(frame) {
    if (!initialized) return;
    if (frame.ue_positions) updateUEs(frame.ue_positions);
    if (frame.bs_positions && frame.prb_maps && frame.slices && frame.ue_positions)
      updateConnections(frame.bs_positions, frame.prb_maps, frame.slices, frame.ue_positions);
    if (frame.bs_positions && Object.keys(bsMeshes).length === 0)
      buildBSTowers(frame.bs_positions);
    updatePRBDisplay(frame.slices || [], frame.prb_maps || {});
  }

  function getServiceLabel(type, sid) {
    const list = {
      eMBB:  ["4K Video Streaming", "High-Speed Web", "Mobile Cloud Gaming"],
      URLLC: ["Remote Surgery V2X", "Autonomous Driving", "Grid Smart Meters"],
      mMTC:  ["Smart City Sensors", "Industrial IoT", "Wearable Telemetry"]
    }[type] || ["General Data"];
    return list[sid % list.length];
  }

  function updatePRBDisplay(slices, prbMaps) {
    const el = document.getElementById('prb-display');
    if (!el) return;
    const bsMap = prbMaps['0'] || prbMaps[0] || {};
    el.innerHTML = slices.map(sl => {
      const p = bsMap[sl.slice_id] || bsMap[String(sl.slice_id)] || 0;
      const pct = Math.min(p, 100);
      const col = { eMBB:'#0088ff', URLLC:'#ff4466', mMTC:'#00ff88' }[sl.slice_type] || '#888';
      const label = getServiceLabel(sl.slice_type, sl.slice_id);
      
      return `<div class="prb-row">
        <span class="prb-label" style="width:115px; font-size:10px;">${label}</span>
        <span class="prb-val">${p}</span>
        <div class="prb-bar"><div class="prb-fill" style="width:${pct}%;background:${col}"></div></div>
      </div>`;
    }).join('');
  }

  function onResize() {
    const c = document.getElementById('scene-container');
    if (!c || !renderer) return;
    camera.aspect = c.clientWidth / c.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(c.clientWidth, c.clientHeight);
  }

  return { init, update, onResize };
})();
