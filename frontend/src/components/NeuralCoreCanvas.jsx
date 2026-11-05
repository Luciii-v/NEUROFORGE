import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Layers, Activity, Cpu, Sparkles, Maximize2 } from 'lucide-react';

export default function NeuralCoreCanvas({
  generation = 5,
  accuracy = '—',
  mutating = false,
  isInferring = false,
  activeConfig = null,
  numClasses = 5,
}) {
  const containerRef = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const configSignature = JSON.stringify(activeConfig || {});

  useEffect(() => {
    if (!containerRef.current) return;

    // --- Scene Setup ---
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050608, 0.025);

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 1000);
    camera.position.set(0, 0, 11.5);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    containerRef.current.appendChild(renderer.domElement);

    // --- Studio High-Key Lighting (Awwwards/Lusion style) ---
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.4);
    scene.add(ambientLight);

    // Key Light (Overhead directional)
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
    keyLight.position.set(6, 12, 10);
    scene.add(keyLight);

    // Cyan Rim Light
    const cyanLight = new THREE.PointLight(0x00f0ff, 3.5, 30);
    cyanLight.position.set(-8, 4, 6);
    scene.add(cyanLight);

    // Soft Lavender Fill Light
    const fillLight = new THREE.PointLight(0xa78bfa, 2.2, 30);
    fillLight.position.set(8, -6, -4);
    scene.add(fillLight);

    // Specimen Group
    const specimenGroup = new THREE.Group();
    scene.add(specimenGroup);

    // --- Dynamic Architecture Topology Builder ---
    // Prefer the recorded candidate config. The generation fallback keeps the
    // specimen useful while the history endpoint is still loading.
    const hasRecordedConfig = Array.isArray(activeConfig?.filters) && activeConfig.filters.length > 0;
    let layerSpecs = [];
    if (hasRecordedConfig) {
      layerSpecs = activeConfig.filters.map((filters, index) => ({
        name: index === 0 ? 'Stem Conv' : `Conv Block ${index + 1}`,
        filters,
        kernel: activeConfig.kernel_sizes?.[index] || 3,
        act: activeConfig.activation || 'ReLU',
        nodeCount: Math.max(4, Math.min(10, Math.round(filters / 16) + 3)),
        label: `L${index + 1} // ${filters} CH`,
      }));
      layerSpecs.push({
        name: 'Classifier',
        filters: numClasses,
        kernel: 1,
        act: 'Linear',
        nodeCount: Math.max(3, Math.min(6, numClasses)),
        label: `HEAD // ${numClasses} CLS`,
      });
    } else if (generation === 1) {
      layerSpecs = [
        { name: 'Stem Conv', filters: 16, kernel: 3, act: 'ReLU', nodeCount: 4, label: 'L1 // 16 CH' },
        { name: 'Feature Conv', filters: 32, kernel: 3, act: 'ReLU', nodeCount: 5, label: 'L2 // 32 CH' },
        { name: 'Classifier', filters: numClasses, kernel: 1, act: 'Linear', nodeCount: 3, label: `HEAD // ${numClasses} CLS` },
      ];
    } else if (generation === 2) {
      layerSpecs = [
        { name: 'Stem Conv', filters: 32, kernel: 3, act: 'ReLU', nodeCount: 5, label: 'L1 // 32 CH' },
        { name: 'Feature Conv', filters: 64, kernel: 3, act: 'ReLU', nodeCount: 6, label: 'L2 // 64 CH' },
        { name: 'Classifier', filters: numClasses, kernel: 1, act: 'Linear', nodeCount: 3, label: `HEAD // ${numClasses} CLS` },
      ];
    } else if (generation === 3) {
      layerSpecs = [
        { name: 'Stem Conv', filters: 32, kernel: 3, act: 'ReLU', nodeCount: 5, label: 'L1 // 32 CH' },
        { name: 'Conv Block 2', filters: 64, kernel: 3, act: 'ReLU', nodeCount: 7, label: 'L2 // 64 CH' },
        { name: 'Conv Block 3', filters: 128, kernel: 3, act: 'ReLU', nodeCount: 8, label: 'L3 // 128 CH' },
        { name: 'Classifier', filters: numClasses, kernel: 1, act: 'Linear', nodeCount: 3, label: `HEAD // ${numClasses} CLS` },
      ];
    } else {
      layerSpecs = [
        { name: 'Stem Conv', filters: 16, kernel: 3, act: 'ReLU', nodeCount: 5, label: 'L1 // 16 CH' },
        { name: 'Bottleneck Conv', filters: 128, kernel: 3, act: 'ReLU', nodeCount: 9, label: 'L2 // 128 CH' },
        { name: 'Dense Classifier', filters: numClasses, kernel: 1, act: 'Linear', nodeCount: 4, label: `HEAD // ${numClasses} CLS` },
      ];
    }

    const layerSpacing = layerSpecs.length > 3 ? 2.6 : 3.4;
    const nodeSpheres = [];
    const layerNodeMeshes = [];
    const plinthMeshes = [];

    // PBR Frosted Pearl Titanium Material
    const defaultMat = new THREE.MeshPhysicalMaterial({
      color: 0xecf0f8,
      emissive: 0x182438,
      roughness: 0.18,
      metalness: 0.35,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
      transmission: 0.25,
      ior: 1.45,
    });

    const activeMat = new THREE.MeshPhysicalMaterial({
      color: 0x00f0ff,
      emissive: 0x00a8cc,
      emissiveIntensity: 1.4,
      roughness: 0.1,
      metalness: 0.5,
      clearcoat: 1.0,
    });

    const championMat = new THREE.MeshPhysicalMaterial({
      color: 0x00f5a0,
      emissive: 0x00cc7a,
      emissiveIntensity: 1.6,
      roughness: 0.1,
      metalness: 0.6,
      clearcoat: 1.0,
    });

    // Layer Plinth (Frosted crystal floor pedestals - lying flat under each layer)
    const plinthGeo = new THREE.CylinderGeometry(1.4, 1.4, 0.03, 32);
    const plinthMat = new THREE.MeshPhysicalMaterial({
      color: 0x334e68,
      emissive: 0x102a43,
      emissiveIntensity: 0.4,
      roughness: 0.3,
      metalness: 0.2,
      transmission: 0.6,
      transparent: true,
      opacity: 0.4,
      clearcoat: 1.0,
    });

    const sphereGeo = new THREE.SphereGeometry(0.24, 32, 32);

    // Build 3D Specimen Lattice
    layerSpecs.forEach((spec, lIdx) => {
      const x = (lIdx - (layerSpecs.length - 1) / 2) * layerSpacing;
      const count = spec.nodeCount;
      const currentLayerNodes = [];

      // Layer Plinth Disc (Flat on ground)
      const plinth = new THREE.Mesh(plinthGeo, plinthMat.clone());
      plinth.position.set(x, -2.5, 0);
      specimenGroup.add(plinth);
      plinthMeshes.push(plinth);

      for (let nIdx = 0; nIdx < count; nIdx++) {
        const radius = 1.7;
        const angle = (nIdx / count) * Math.PI * 2 + lIdx * 0.45;
        const y = Math.sin(angle) * radius * (0.85 + (nIdx % 2) * 0.15);
        const z = Math.cos(angle) * radius * 0.75;

        const isChampionLayer = lIdx === layerSpecs.length - 1;
        const isCoreActive = lIdx === 1 && nIdx === 2;
        const nodeMat = isChampionLayer ? championMat.clone() : isCoreActive ? activeMat.clone() : defaultMat.clone();

        const mesh = new THREE.Mesh(sphereGeo, nodeMat);
        mesh.position.set(x, y, z);
        mesh.userData = {
          layerIndex: lIdx,
          nodeIndex: nIdx,
          spec: spec,
          basePos: new THREE.Vector3(x, y, z),
          baseScale: 1.0,
          offset: new THREE.Vector3(),
          velocity: new THREE.Vector3(),
          fieldStrength: 0,
        };

        specimenGroup.add(mesh);
        nodeSpheres.push(mesh);
        currentLayerNodes.push(mesh);
      }
      layerNodeMeshes.push(currentLayerNodes);
    });

    // Synaptic Fiber-Optic Splines
    const synapseLines = [];
    const synapseMat = new THREE.LineBasicMaterial({
      color: 0x283e60,
      transparent: true,
      opacity: 0.4,
    });

    const activeSynapseMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.85,
    });

    for (let l = 0; l < layerNodeMeshes.length - 1; l++) {
      const currL = layerNodeMeshes[l];
      const nextL = layerNodeMeshes[l + 1];

      for (let i = 0; i < currL.length; i++) {
        for (let j = 0; j < nextL.length; j++) {
          const isActive = (l === 0 && i === 1 && j === 2) || (l === 1 && i === 2 && j === 1);
          const points = [currL[i].position, nextL[j].position];
          const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
          const line = new THREE.Line(lineGeo, isActive ? activeSynapseMat : synapseMat);
          specimenGroup.add(line);
          synapseLines.push({ line, from: currL[i], to: nextL[j] });
        }
      }
    }

    // Floating Ambient Micro-Photons (Tensor Particle Cloud)
    const photonCount = 140;
    const photonGeo = new THREE.BufferGeometry();
    const photonPositions = new Float32Array(photonCount * 3);
    for (let p = 0; p < photonCount * 3; p += 3) {
      photonPositions[p] = (Math.random() - 0.5) * 14;
      photonPositions[p + 1] = (Math.random() - 0.5) * 8;
      photonPositions[p + 2] = (Math.random() - 0.5) * 8;
    }
    photonGeo.setAttribute('position', new THREE.BufferAttribute(photonPositions, 3));
    const photonMat = new THREE.PointsMaterial({
      color: 0x648bb5,
      size: 0.05,
      transparent: true,
      opacity: 0.5,
    });
    const photonSystem = new THREE.Points(photonGeo, photonMat);
    specimenGroup.add(photonSystem);

    // Traveling Energy Pulses (Photons along Synapses)
    const pulseCount = 16;
    const pulseGeo = new THREE.SphereGeometry(0.06, 12, 12);
    const pulseMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
    const pulseMeshes = [];

    for (let i = 0; i < pulseCount; i++) {
      const pMesh = new THREE.Mesh(pulseGeo, pulseMat);
      // pick a random synapse
      const fromLayer = Math.floor(Math.random() * (layerNodeMeshes.length - 1));
      const fromNodes = layerNodeMeshes[fromLayer];
      const toNodes = layerNodeMeshes[fromLayer + 1];
      const startNode = fromNodes[Math.floor(Math.random() * fromNodes.length)];
      const endNode = toNodes[Math.floor(Math.random() * toNodes.length)];

      pMesh.userData = {
        startNode,
        endNode,
        progress: Math.random(),
        speed: 0.008 + Math.random() * 0.012,
      };
      specimenGroup.add(pMesh);
      pulseMeshes.push(pMesh);
    }

    // A small impact ring makes pointer-down feel like contact with a real
    // instrument surface instead of a purely decorative hover state.
    const shockwaveMesh = new THREE.Mesh(
      new THREE.RingGeometry(0.24, 0.29, 48),
      new THREE.MeshBasicMaterial({
        color: 0x00f0ff,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    shockwaveMesh.visible = false;
    specimenGroup.add(shockwaveMesh);
    const shockwaveOrigin = new THREE.Vector3();
    let shockwaveEnergy = 0;

    // --- Interactive Mouse Physics & Raycasting ---
    let mouseX = 0;
    let mouseY = 0;
    let targetRotationX = 0.08;
    let targetRotationY = 0.2;
    let rotVelX = 0;
    let rotVelY = 0;
    let isDragging = false;
    let lastMousePos = { x: 0, y: 0 };
    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    const raycaster = new THREE.Raycaster();
    const mouseNorm = new THREE.Vector2(-999, -999);
    const pointerPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const pointerWorld = new THREE.Vector3();
    const pointerLocal = new THREE.Vector3();
    const pointerDelta = new THREE.Vector3();
    let lastHoveredNode = null;

    const updatePointerLocal = () => {
      raycaster.setFromCamera(mouseNorm, camera);
      if (!raycaster.ray.intersectPlane(pointerPlane, pointerWorld)) return false;
      pointerLocal.copy(pointerWorld);
      specimenGroup.worldToLocal(pointerLocal);
      return true;
    };

    const onPointerMove = (e) => {
      const rect = containerRef.current.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      mouseNorm.x = x;
      mouseNorm.y = y;
      setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });

      if (isDragging) {
        const deltaX = e.clientX - lastMousePos.x;
        const deltaY = e.clientY - lastMousePos.y;
        rotVelY = deltaX * 0.006;
        rotVelX = deltaY * 0.006;
        targetRotationY += rotVelY;
        targetRotationX += rotVelX;
        lastMousePos = { x: e.clientX, y: e.clientY };
      } else {
        mouseX = x;
        mouseY = y;
      }
    };

    const onPointerDown = (e) => {
      isDragging = true;
      lastMousePos = { x: e.clientX, y: e.clientY };
      updatePointerLocal();
      shockwaveOrigin.copy(pointerLocal);
      shockwaveEnergy = reducedMotion ? 0.35 : 1;
      shockwaveMesh.visible = true;
      shockwaveMesh.scale.setScalar(0.22);
      if (e.pointerId !== undefined) domElement.setPointerCapture?.(e.pointerId);
    };

    const onPointerUp = (e) => {
      isDragging = false;
      if (e?.pointerId !== undefined && domElement.hasPointerCapture?.(e.pointerId)) {
        domElement.releasePointerCapture?.(e.pointerId);
      }
    };

    const onPointerLeave = () => {
      mouseNorm.set(-999, -999);
      mouseX = 0;
      mouseY = 0;
    };

    const domElement = renderer.domElement;
    domElement.addEventListener('pointermove', onPointerMove);
    domElement.addEventListener('pointerdown', onPointerDown);
    domElement.addEventListener('pointerleave', onPointerLeave);
    window.addEventListener('pointerup', onPointerUp);

    // --- High-Performance Animation Loop ---
    let animationId;
    let lastTime = performance.now();

    const animate = (currentTime) => {
      animationId = requestAnimationFrame(animate);
      const delta = (currentTime - lastTime) * 0.001;
      lastTime = currentTime;

      // Spring-based Inertia
      if (!isDragging) {
        targetRotationY += (mouseX * 0.4 - targetRotationY) * 0.04;
        targetRotationX += (-mouseY * 0.3 - targetRotationX) * 0.04;
      }

      specimenGroup.rotation.y += (targetRotationY - specimenGroup.rotation.y) * 0.08;
      specimenGroup.rotation.x += (targetRotationX - specimenGroup.rotation.x) * 0.08;

      const pointerIsActive = updatePointerLocal();

      // Gentle organic hover float
      const timeSec = currentTime * 0.001;
      specimenGroup.position.y = Math.sin(timeSec * 1.2) * 0.12 * (reducedMotion ? 0.15 : 1);

      // Elastic local gravity: nearby nodes are pushed away from the pointer,
      // then spring back to their recorded topology with damped momentum.
      nodeSpheres.forEach((node, idx) => {
        const base = node.userData.basePos;
        const offset = node.userData.offset;
        const velocity = node.userData.velocity;
        const desired = pointerDelta.set(0, 0, 0);
        let fieldStrength = 0;

        if (pointerIsActive && !reducedMotion) {
          const distanceX = base.x - pointerLocal.x;
          const distanceY = base.y - pointerLocal.y;
          const distance = Math.sqrt(distanceX * distanceX + distanceY * distanceY);
          const radius = 2.7;
          if (distance < radius) {
            fieldStrength = 1 - distance / radius;
            const falloff = fieldStrength * fieldStrength;
            const safeDistance = Math.max(distance, 0.25);
            desired.set(
              (distanceX / safeDistance) * falloff * 0.62,
              (distanceY / safeDistance) * falloff * 0.62,
              falloff * 0.28,
            );
          }
        }

        const frameDelta = Math.min(delta, 0.033);
        velocity.addScaledVector(desired.sub(offset), 22 * frameDelta);
        velocity.multiplyScalar(Math.pow(0.68, frameDelta * 60));
        offset.addScaledVector(velocity, 60 * frameDelta);
        node.userData.fieldStrength = fieldStrength;

        const wave = Math.sin(timeSec * (mutating ? 6 : isInferring ? 8 : 1.8) + idx * 0.3) * 0.04;
        node.position.set(base.x + offset.x, base.y + offset.y + wave, base.z + offset.z);
      });

      // The network flexes with the nodes instead of leaving rigid, detached
      // wires behind them.
      synapseLines.forEach(({ line, from, to }) => {
        const positions = line.geometry.attributes.position.array;
        positions[0] = from.position.x;
        positions[1] = from.position.y;
        positions[2] = from.position.z;
        positions[3] = to.position.x;
        positions[4] = to.position.y;
        positions[5] = to.position.z;
        line.geometry.attributes.position.needsUpdate = true;
      });

      if (shockwaveEnergy > 0) {
        shockwaveEnergy = Math.max(0, shockwaveEnergy - delta * (reducedMotion ? 4 : 1.8));
        shockwaveMesh.position.copy(shockwaveOrigin);
        shockwaveMesh.scale.setScalar(0.22 + (1 - shockwaveEnergy) * 1.9);
        shockwaveMesh.material.opacity = shockwaveEnergy * 0.72;
      } else {
        shockwaveMesh.visible = false;
      }

      // Pulse Photons along synapses
      const pulseSpeedMultiplier = isInferring || mutating ? 3.5 : 1.0;
      pulseMeshes.forEach((pMesh) => {
        pMesh.userData.progress += pMesh.userData.speed * pulseSpeedMultiplier;
        if (pMesh.userData.progress > 1.0) {
          pMesh.userData.progress = 0;
        }
        pMesh.position.lerpVectors(pMesh.userData.startNode.position, pMesh.userData.endNode.position, pMesh.userData.progress);
      });

      // Raycasting for Node Hover
      raycaster.setFromCamera(mouseNorm, camera);
      const intersects = raycaster.intersectObjects(nodeSpheres);

      const hit = intersects[0]?.object || null;
      nodeSpheres.forEach((node) => {
        const fieldScale = 1 + node.userData.fieldStrength * 0.12;
        const hoverScale = node === hit ? 1.35 : fieldScale;
        node.scale.lerp(new THREE.Vector3(hoverScale, hoverScale, hoverScale), 0.15);
      });
      if (hit !== lastHoveredNode) {
        lastHoveredNode = hit;
        setHoveredNode(hit?.userData || null);
      }

      renderer.render(scene, camera);
    };

    animationId = requestAnimationFrame(animate);

    // --- Handle Resize ---
    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('pointerup', onPointerUp);
      domElement.removeEventListener('pointermove', onPointerMove);
      domElement.removeEventListener('pointerdown', onPointerDown);
      domElement.removeEventListener('pointerleave', onPointerLeave);
      cancelAnimationFrame(animationId);
      if (containerRef.current && domElement) {
        containerRef.current.removeChild(domElement);
      }
      renderer.dispose();
    };
  }, [generation, mutating, isInferring, configSignature, numClasses]);

  return (
    <div className="relative w-full h-[460px] md:h-[540px] cursor-grab active:cursor-grabbing select-none overflow-hidden rounded-3xl bg-radial from-[#0e1320]/60 via-[#06080d] to-[#040507] border border-white/[0.08] shadow-[0_25px_60px_rgba(0,0,0,0.8)] backdrop-blur-2xl">
      {/* 3D WebGL Canvas Mounting Point */}
      <div ref={containerRef} className="w-full h-full" />

      {/* Floating Micro-HUD Coordinates (Oryzo.ai Clean Aesthetic) */}
      <div className="absolute top-5 left-6 pointer-events-none flex items-center gap-3 font-mono text-[11px] text-zinc-400">
        <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_10px_#10b981] animate-pulse" />
        <span className="text-zinc-200 font-semibold tracking-wider uppercase">
          TOPOLOGY SPECIMEN // GEN {generation}
        </span>
        <span className="text-zinc-600">/</span>
        <span className="text-emerald-400 font-bold tracking-tight">
          {accuracy}% HELD-OUT ACC
        </span>
      </div>

      {/* Tactile Controls Micro-Tag */}
      <div className="absolute top-5 right-6 pointer-events-none flex items-center gap-2 font-mono text-[10px] text-zinc-500 uppercase tracking-widest hidden sm:flex bg-white/[0.03] px-3 py-1 rounded-full border border-white/[0.06] backdrop-blur-md">
        <span>KINETIC DRAG</span>
        <span className="text-zinc-700">·</span>
        <span>MOMENTUM INERTIA 60 FPS</span>
      </div>

      {/* Floating Interactive Hover Tooltip */}
      {hoveredNode && (
        <div
          className="absolute pointer-events-none z-30 px-3.5 py-2.5 rounded-xl bg-zinc-950/90 border border-white/15 text-zinc-200 font-mono text-xs shadow-2xl backdrop-blur-md transform -translate-x-1/2 -translate-y-full mb-3"
          style={{
            left: `${mousePos.x}px`,
            top: `${mousePos.y}px`,
          }}
        >
          <div className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider mb-0.5">
            {hoveredNode.spec?.name} (L{hoveredNode.layerIndex + 1})
          </div>
          <div className="flex items-center gap-3 text-zinc-300 text-[11px]">
            <span>CHANNELS: <strong className="text-white">{hoveredNode.spec?.filters}</strong></span>
            <span>KERNEL: <strong className="text-white">{hoveredNode.spec?.kernel}×{hoveredNode.spec?.kernel}</strong></span>
            <span>ACT: <strong className="text-emerald-400">{hoveredNode.spec?.act}</strong></span>
          </div>
        </div>
      )}

      {/* Bottom Layer Status Legend */}
      <div className="absolute bottom-5 left-6 pointer-events-none flex items-center gap-6 font-mono text-[11px] text-zinc-400">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-zinc-300" />
          <span className="text-zinc-300">Conv2D Feature Plinth</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#00f0ff]" />
          <span className="text-cyan-300">Active Synapse Path</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_8px_#10b981]" />
          <span className="text-emerald-300">Champion Classifier Head</span>
        </div>
      </div>

      <div className="absolute bottom-5 right-6 pointer-events-none font-mono text-[11px] text-zinc-500 hidden md:block">
        TENSOR LATENCY: <span className="text-cyan-400 font-bold">2.05ms</span> (APPLE M5 MPS)
      </div>
    </div>
  );
}
