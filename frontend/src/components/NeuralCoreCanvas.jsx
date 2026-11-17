import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { Layers, Activity, Cpu, Sparkles, Zap, Award } from 'lucide-react';
import soundEngine from '../utils/soundEngine';

export default function NeuralCoreCanvas({
  generation = 5,
  accuracy = '—',
  mutating = false,
  isInferring = false,
  activeConfig = null,
  numClasses = 5,
  latencyMs = null,
  lastInference = null,
}) {
  const containerRef = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [waveActive, setWaveActive] = useState(false);
  const [winnerInfo, setWinnerInfo] = useState(null);
  const configSignature = JSON.stringify(activeConfig || {});

  // Refs to decouple Three.js render loop from React state updates
  const triggerPulseRef = useRef(null);
  const isInferringRef = useRef(isInferring);
  const mutatingRef = useRef(mutating);
  const lastInferenceRef = useRef(lastInference);

  useEffect(() => {
    isInferringRef.current = isInferring;
    if (isInferring && triggerPulseRef.current) {
      triggerPulseRef.current();
    }
  }, [isInferring]);

  useEffect(() => {
    mutatingRef.current = mutating;
    if (mutating) {
      soundEngine.mutate();
    }
  }, [mutating]);

  useEffect(() => {
    lastInferenceRef.current = lastInference;
    if (lastInference) {
      setWinnerInfo({
        prediction: lastInference.prediction,
        percentage: lastInference.percentage,
        latency: lastInference.latency_ms,
        modality: lastInference.modality,
      });
      if (triggerPulseRef.current) {
        triggerPulseRef.current(lastInference);
      }
    }
  }, [lastInference]);

  useEffect(() => {
    if (!containerRef.current) return;

    // --- Scene Setup ---
    const width = containerRef.current.clientWidth || 860;
    const height = containerRef.current.clientHeight || 520;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050608, 0.006);

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 1000);
    camera.position.set(0, 0, 10.2);

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance',
      });
    } catch (error) {
      // No WebGL (old GPU, remote desktop, locked-down browser): keep the rest of the studio alive.
      console.warn('3D specimen disabled:', error.message);
      containerRef.current.dataset.webgl = 'unavailable';
      return undefined;
    }
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    containerRef.current.appendChild(renderer.domElement);

    // --- Studio High-Key Directional Lighting ---
    const ambientLight = new THREE.AmbientLight(0xffffff, 2.2);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
    keyLight.position.set(6, 12, 10);
    scene.add(keyLight);

    const cyanLight = new THREE.PointLight(0x00f0ff, 4.0, 35);
    cyanLight.position.set(-8, 4, 6);
    scene.add(cyanLight);

    const fillLight = new THREE.PointLight(0xa78bfa, 2.8, 35);
    fillLight.position.set(8, -6, -4);
    scene.add(fillLight);

    // Dynamic wave-glow light that travels through the network
    const waveLight = new THREE.PointLight(0x00f5a0, 0, 15);
    scene.add(waveLight);

    // Specimen Group
    const specimenGroup = new THREE.Group();
    scene.add(specimenGroup);

    // --- Dynamic Architecture Topology Builder ---
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
        name: 'Classifier Head',
        filters: numClasses,
        kernel: 1,
        act: 'Softmax',
        nodeCount: Math.max(3, Math.min(6, numClasses)),
        label: `HEAD // ${numClasses} CLS`,
      });
    } else if (generation === 1) {
      layerSpecs = [
        { name: 'Stem Conv', filters: 16, kernel: 3, act: 'ReLU', nodeCount: 4, label: 'L1 // 16 CH' },
        { name: 'Feature Conv', filters: 32, kernel: 3, act: 'ReLU', nodeCount: 5, label: 'L2 // 32 CH' },
        { name: 'Classifier Head', filters: numClasses, kernel: 1, act: 'Softmax', nodeCount: 3, label: `HEAD // ${numClasses} CLS` },
      ];
    } else if (generation === 2) {
      layerSpecs = [
        { name: 'Stem Conv', filters: 32, kernel: 3, act: 'ReLU', nodeCount: 5, label: 'L1 // 32 CH' },
        { name: 'Feature Conv', filters: 64, kernel: 3, act: 'ReLU', nodeCount: 6, label: 'L2 // 64 CH' },
        { name: 'Classifier Head', filters: numClasses, kernel: 1, act: 'Softmax', nodeCount: 3, label: `HEAD // ${numClasses} CLS` },
      ];
    } else if (generation === 3) {
      layerSpecs = [
        { name: 'Stem Conv', filters: 32, kernel: 3, act: 'ReLU', nodeCount: 5, label: 'L1 // 32 CH' },
        { name: 'Conv Block 2', filters: 64, kernel: 3, act: 'ReLU', nodeCount: 7, label: 'L2 // 64 CH' },
        { name: 'Conv Block 3', filters: 128, kernel: 3, act: 'ReLU', nodeCount: 8, label: 'L3 // 128 CH' },
        { name: 'Classifier Head', filters: numClasses, kernel: 1, act: 'Softmax', nodeCount: 4, label: `HEAD // ${numClasses} CLS` },
      ];
    } else {
      layerSpecs = [
        { name: 'Stem Conv', filters: 16, kernel: 3, act: 'ReLU', nodeCount: 5, label: 'L1 // 16 CH' },
        { name: 'Bottleneck Conv', filters: 128, kernel: 3, act: 'ReLU', nodeCount: 8, label: 'L2 // 128 CH' },
        { name: 'Classifier Head', filters: numClasses, kernel: 1, act: 'Softmax', nodeCount: 4, label: `HEAD // ${numClasses} CLS` },
      ];
    }

    const layerSpacing = layerSpecs.length > 3 ? 2.6 : 3.4;
    const nodeSpheres = [];
    const layerNodeMeshes = [];
    const plinthMeshes = [];

    // Base Materials - PBR Frosted Pearl Titanium with emissive core (Safari & WebGL cross-compatible)
    const defaultMat = new THREE.MeshPhysicalMaterial({
      color: 0xf1f5f9,
      emissive: 0x38bdf8,
      emissiveIntensity: 0.55,
      roughness: 0.15,
      metalness: 0.45,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
    });

    const activeMat = new THREE.MeshPhysicalMaterial({
      color: 0x00f0ff,
      emissive: 0x00d4ff,
      emissiveIntensity: 1.8,
      roughness: 0.1,
      metalness: 0.6,
      clearcoat: 1.0,
    });

    const championMat = new THREE.MeshPhysicalMaterial({
      color: 0x10b981,
      emissive: 0x059669,
      emissiveIntensity: 2.2,
      roughness: 0.1,
      metalness: 0.6,
      clearcoat: 1.0,
    });

    // Layer Plinth (Frosted crystal floor pedestals)
    const plinthGeo = new THREE.CylinderGeometry(1.4, 1.4, 0.03, 32);
    const plinthMat = new THREE.MeshPhysicalMaterial({
      color: 0x1e293b,
      emissive: 0x0f172a,
      emissiveIntensity: 0.5,
      roughness: 0.25,
      metalness: 0.3,
      transparent: true,
      opacity: 0.6,
      clearcoat: 1.0,
    });

    const sphereGeo = new THREE.SphereGeometry(0.36, 32, 32);

    // Build 3D Specimen Lattice
    layerSpecs.forEach((spec, lIdx) => {
      const x = (lIdx - (layerSpecs.length - 1) / 2) * layerSpacing;
      const count = spec.nodeCount;
      const currentLayerNodes = [];

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
          pulseBoost: 0, // dynamic energy spike
          baseEmissive: isChampionLayer ? 1.6 : isCoreActive ? 1.4 : 0.35,
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
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.5,
    });

    const activeSynapseMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.95,
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
          synapseLines.push({ line, from: currL[i], to: nextL[j], layerIndex: l });
        }
      }
    }

    // Ambient Micro-Photons (Background floating tensor cloud)
    const photonCount = 120;
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
      opacity: 0.45,
    });
    const photonSystem = new THREE.Points(photonGeo, photonMat);
    specimenGroup.add(photonSystem);

    // --- HIGH-ENERGY SYNAPTIC LIGHT PULSE SYSTEM ---
    // High-speed photons that cascade through the network during forward passes
    const cascadePhotonCount = 36;
    const cascadeGeo = new THREE.SphereGeometry(0.08, 16, 16);
    const cascadeMat = new THREE.MeshBasicMaterial({
      color: 0x00f5a0,
      transparent: true,
      opacity: 0,
    });
    const cascadePhotons = [];

    for (let i = 0; i < cascadePhotonCount; i++) {
      const cMesh = new THREE.Mesh(cascadeGeo, cascadeMat.clone());
      cMesh.visible = false;
      cMesh.userData = {
        layerIndex: 0,
        fromNode: null,
        toNode: null,
        progress: 0,
        speed: 1.4,
        active: false,
      };
      specimenGroup.add(cMesh);
      cascadePhotons.push(cMesh);
    }

    // Interactive pointer shockwave ring
    const shockwaveMesh = new THREE.Mesh(
      new THREE.RingGeometry(0.24, 0.32, 48),
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

    // Champion Output Shockwave Ring (fires at winning classification head node)
    const winnerShockwave = new THREE.Mesh(
      new THREE.RingGeometry(0.28, 0.38, 48),
      new THREE.MeshBasicMaterial({
        color: 0x00f5a0,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    winnerShockwave.visible = false;
    specimenGroup.add(winnerShockwave);
    let winnerShockwaveEnergy = 0;
    const winnerShockwaveOrigin = new THREE.Vector3();

    // Wave Progression State
    const waveState = {
      active: false,
      progress: 0,
      duration: 1.15, // seconds for full forward pass
      lastSoundLayer: -1,
      targetClassIdx: 0,
    };

    // Public method to trigger the synaptic light cascade
    const triggerSynapticWave = (inferenceData = null) => {
      waveState.active = true;
      waveState.progress = 0;
      waveState.lastSoundLayer = -1;
      setWaveActive(true);

      // Determine target winning node on the classifier head
      const headNodes = layerNodeMeshes[layerNodeMeshes.length - 1] || [];
      if (inferenceData?.all_classes && inferenceData?.prediction) {
        const idx = inferenceData.all_classes.indexOf(inferenceData.prediction);
        waveState.targetClassIdx = idx >= 0 ? idx % headNodes.length : 0;
      } else {
        waveState.targetClassIdx = Math.floor(Math.random() * Math.max(1, headNodes.length));
      }

      // Initial sound sweep
      soundEngine.inferStart();

      // Spawn cascading photons along synapses from input layer
      cascadePhotons.forEach((p, idx) => {
        const fromLayer = idx % (layerNodeMeshes.length - 1);
        const fromNodes = layerNodeMeshes[fromLayer];
        const toNodes = layerNodeMeshes[fromLayer + 1];
        p.userData.fromNode = fromNodes[Math.floor(Math.random() * fromNodes.length)];
        p.userData.toNode = toNodes[Math.floor(Math.random() * toNodes.length)];
        p.userData.layerIndex = fromLayer;
        p.userData.progress = (idx % 4) * 0.1;
        p.userData.active = true;
        p.visible = true;
        p.material.opacity = 0.9;
      });
    };

    triggerPulseRef.current = triggerSynapticWave;

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
      soundEngine.tick(900, 0.015);
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

    // --- High-Performance 60FPS Animation Loop ---
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
      const timeSec = currentTime * 0.001;
      specimenGroup.position.y = Math.sin(timeSec * 1.2) * 0.12 * (reducedMotion ? 0.15 : 1);

      // --- SYNAPTIC FORWARD PASS WAVE PROPAGATION ---
      if (waveState.active) {
        waveState.progress += delta / waveState.duration;
        const totalL = layerNodeMeshes.length;
        const currentLayerIdx = Math.floor(waveState.progress * totalL);

        // Position dynamic point light along the forward pass
        const startX = -((totalL - 1) / 2) * layerSpacing;
        const currentWaveX = startX + waveState.progress * (totalL - 1) * layerSpacing;
        waveLight.position.set(currentWaveX, 0, 1.5);
        waveLight.intensity = Math.sin(waveState.progress * Math.PI) * 4.5;

        // Sound trigger and node ignition as wave crosses each layer
        if (currentLayerIdx !== waveState.lastSoundLayer && currentLayerIdx < totalL) {
          waveState.lastSoundLayer = currentLayerIdx;
          soundEngine.synapsePulse(currentLayerIdx, totalL);

          // Ignite nodes in the reached layer
          const layerNodes = layerNodeMeshes[currentLayerIdx] || [];
          layerNodes.forEach((node) => {
            node.userData.pulseBoost = 1.0;
          });
        }

        // Animate cascade photons along active forward path
        cascadePhotons.forEach((pMesh) => {
          if (!pMesh.userData.active || !pMesh.userData.fromNode || !pMesh.userData.toNode) return;
          pMesh.userData.progress += delta * pMesh.userData.speed;
          if (pMesh.userData.progress <= 1.0) {
            pMesh.position.lerpVectors(
              pMesh.userData.fromNode.position,
              pMesh.userData.toNode.position,
              pMesh.userData.progress
            );
          } else {
            pMesh.userData.progress = 1.0;
            pMesh.material.opacity = Math.max(0, pMesh.material.opacity - delta * 3);
          }
        });

        // Wave resolution at Classifier Head
        if (waveState.progress >= 1.0) {
          waveState.active = false;
          setWaveActive(false);

          const headNodes = layerNodeMeshes[layerNodeMeshes.length - 1] || [];
          const winnerNode = headNodes[waveState.targetClassIdx % headNodes.length] || headNodes[0];

          if (winnerNode) {
            winnerNode.userData.pulseBoost = 2.2;
            winnerShockwaveOrigin.copy(winnerNode.position);
            winnerShockwave.position.copy(winnerNode.position);
            winnerShockwaveEnergy = 1.0;
            winnerShockwave.visible = true;
          }

          soundEngine.championChord(true);
        }
      }

      // Fade out champion shockwave ring
      if (winnerShockwaveEnergy > 0) {
        winnerShockwaveEnergy = Math.max(0, winnerShockwaveEnergy - delta * 1.8);
        winnerShockwave.scale.setScalar(0.24 + (1 - winnerShockwaveEnergy) * 2.4);
        winnerShockwave.material.opacity = winnerShockwaveEnergy * 0.9;
      } else {
        winnerShockwave.visible = false;
      }

      // Elastic local gravity & dynamic emissive dissipation on nodes
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

        // Wave motion
        const isCurrentlyInferring = isInferringRef.current || waveState.active;
        const wave = Math.sin(timeSec * (mutatingRef.current ? 6 : isCurrentlyInferring ? 7 : 1.8) + idx * 0.3) * 0.04;
        node.position.set(base.x + offset.x, base.y + offset.y + wave, base.z + offset.z);

        // Exponential energy dissipation after synaptic pulse
        if (node.userData.pulseBoost > 0) {
          node.userData.pulseBoost = Math.max(0, node.userData.pulseBoost - delta * 2.2);
        }

        // Apply dynamic emissive intensity spike
        const extraEmissive = node.userData.pulseBoost * 3.2;
        node.material.emissiveIntensity = node.userData.baseEmissive + extraEmissive;
      });

      // Update synapse lines to track moving nodes
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

      // Interactive pointer shockwave dissipation
      if (shockwaveEnergy > 0) {
        shockwaveEnergy = Math.max(0, shockwaveEnergy - delta * (reducedMotion ? 4 : 1.8));
        shockwaveMesh.position.copy(shockwaveOrigin);
        shockwaveMesh.scale.setScalar(0.22 + (1 - shockwaveEnergy) * 1.9);
        shockwaveMesh.material.opacity = shockwaveEnergy * 0.72;
      } else {
        shockwaveMesh.visible = false;
      }

      // Raycasting for Node Hover
      raycaster.setFromCamera(mouseNorm, camera);
      const intersects = raycaster.intersectObjects(nodeSpheres);
      const hit = intersects[0]?.object || null;

      nodeSpheres.forEach((node) => {
        const fieldScale = 1 + node.userData.fieldStrength * 0.12 + (node.userData.pulseBoost || 0) * 0.28;
        const hoverScale = node === hit ? 1.35 : fieldScale;
        node.scale.lerp(new THREE.Vector3(hoverScale, hoverScale, hoverScale), 0.15);
      });

      if (hit !== lastHoveredNode) {
        if (hit && !lastHoveredNode) {
          soundEngine.hover();
        }
        lastHoveredNode = hit;
        setHoveredNode(hit?.userData || null);
      }

      renderer.render(scene, camera);
    };

    animationId = requestAnimationFrame(animate);

    // --- Handle Resize & Container Sizing ---
    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      if (w > 0 && h > 0) {
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      }
    };

    window.addEventListener('resize', handleResize);

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: w, height: h } = entry.contentRect;
        if (w > 0 && h > 0) {
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }
      }
    });
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
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
  }, [generation, configSignature, numClasses]);

  return (
    <div className="relative w-full h-[460px] md:h-[540px] cursor-grab active:cursor-grabbing select-none overflow-hidden rounded-3xl bg-radial from-[#0e1320]/60 via-[#06080d] to-[#040507] border border-white/[0.08] shadow-[0_25px_60px_rgba(0,0,0,0.8)] backdrop-blur-2xl">
      {/* 3D WebGL Canvas Mounting Point */}
      <div ref={containerRef} className="w-full h-full" />

      {/* Floating Micro-HUD Coordinates */}
      <div className="absolute top-5 left-6 pointer-events-none flex items-center gap-3 font-mono text-[11px] text-zinc-400">
        <span className={`w-2.5 h-2.5 rounded-full ${waveActive ? 'bg-cyan-400 shadow-[0_0_12px_#00f0ff] animate-ping' : 'bg-emerald-400 shadow-[0_0_10px_#10b981]'}`} />
        <span className="text-zinc-200 font-semibold tracking-wider uppercase">
          TOPOLOGY SPECIMEN // GEN {generation}
        </span>
        <span className="text-zinc-600">/</span>
        <span className="text-emerald-400 font-bold tracking-tight">
          {accuracy}% HELD-OUT ACC
        </span>
      </div>

      {/* Manual Pulse Tensor Trigger Button */}
      <div className="absolute top-5 right-6 z-20 flex items-center gap-2">
        <button
          onClick={() => {
            if (triggerPulseRef.current) {
              triggerPulseRef.current();
            }
          }}
          className={`flex items-center gap-1.5 font-mono text-[10px] tracking-wider uppercase px-3 py-1.5 rounded-full border transition-all duration-300 ${
            waveActive
              ? 'bg-cyan-500/20 text-cyan-300 border-cyan-400/40 shadow-[0_0_15px_rgba(0,240,255,0.3)]'
              : 'bg-white/[0.04] hover:bg-white/[0.08] text-zinc-300 border-white/[0.1] hover:border-white/[0.25]'
          }`}
          title="Simulate Synaptic Forward Pass Wave"
        >
          <Zap size={11} className={waveActive ? 'text-cyan-400 fill-cyan-400 animate-bounce' : 'text-zinc-400'} />
          <span>{waveActive ? 'SYNAPSE WAVING...' : 'PULSE TENSOR'}</span>
        </button>
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

      {/* Champion Output Toast Banner (Fires when inference completes) */}
      {winnerInfo && (
        <div className="absolute top-16 left-6 z-20 pointer-events-none animate-in fade-in slide-in-from-top-2 duration-300">
          <div className="flex items-center gap-2.5 bg-emerald-950/60 border border-emerald-500/30 px-3.5 py-1.5 rounded-xl backdrop-blur-xl shadow-lg">
            <Award size={14} className="text-emerald-400" />
            <span className="font-mono text-[10px] text-emerald-200 uppercase tracking-wider">
              RESOLVED CLASSIFIER: <strong className="text-white font-semibold">{winnerInfo.prediction}</strong> ({winnerInfo.percentage}%)
            </span>
            <span className="font-mono text-[10px] text-emerald-400/80">
              · {winnerInfo.latency}ms
            </span>
          </div>
        </div>
      )}

      {/* Bottom Layer Status & Telemetry Bar */}
      <div className="absolute bottom-4 left-6 right-6 pointer-events-none flex flex-wrap items-center justify-between gap-3 font-mono text-[11px]">
        <div className="flex flex-wrap items-center gap-4 text-zinc-400">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-zinc-300" />
            <span className="text-zinc-300">Conv2D Plinth</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${waveActive ? 'bg-cyan-300 shadow-[0_0_12px_#00f0ff]' : 'bg-cyan-400 shadow-[0_0_8px_#00f0ff]'}`} />
            <span className="text-cyan-300">Active Synapse</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#10b981]" />
            <span className="text-emerald-300">Classifier Head</span>
          </div>
        </div>

        <div className="text-zinc-500 font-mono text-[11px] shrink-0">
          TENSOR LATENCY: <span className="text-cyan-400 font-bold">{latencyMs || '—'}</span> (MPS)
        </div>
      </div>
    </div>
  );
}
