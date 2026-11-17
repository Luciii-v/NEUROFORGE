import React, { useState, useEffect, useMemo, useCallback } from 'react';
import NeuralCoreCanvas from './components/NeuralCoreCanvas';
import KineticTrajectoryGraph from './components/KineticTrajectoryGraph';
import InteractiveCrucible from './components/InteractiveCrucible';
import ModelColiseum from './components/ModelColiseum';
import ArchitectureSequencer from './components/ArchitectureSequencer';
import HolographicSpecimenCard from './components/HolographicSpecimenCard';
import CinematicSplash from './components/CinematicSplash';
import MutationLoopPage from './components/MutationLoopPage';
import soundEngine from './utils/soundEngine';
import { LineageBay, TrainingEvidenceBay } from './evidence/EvidencePanels';
import { 
  Play, 
  RotateCcw, 
  Download, 
  Copy, 
  Check, 
  X, 
  ArrowUpRight,
  Layers, 
  FolderOpen,
  Moon,
  Sun,
  Volume2,
  VolumeX,
  Swords,
  Dna,
  Award
} from 'lucide-react';

export default function App() {
  // The title sequence is intentionally part of every fresh page load.
  const [showOpening, setShowOpening] = useState(true);
  const [scrollReady, setScrollReady] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [navScrolled, setNavScrolled] = useState(false);
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('neuroforge-theme') || 'light';
    } catch {
      return 'light';
    }
  });
  const [soundEnabled, setSoundEnabled] = useState(!soundEngine.isMuted);
  const [lastInference, setLastInference] = useState(null);
  const isMutationLoop = new URLSearchParams(window.location.search).get('view') === 'loop';
  const [selectedGen, setSelectedGen] = useState(5);
  const [status, setStatus] = useState(null);
  const [historyPayload, setHistoryPayload] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [mutating, setMutating] = useState(false);
  const [isInferring, setIsInferring] = useState(false);
  const [speech, setSpeech] = useState(
    'The search is ready. Choose a generation, inspect its mutation, and follow the evidence to the current champion.'
  );

  // Drawers
  const [showDatasetDrawer, setShowDatasetDrawer] = useState(false);
  const [showExportDrawer, setShowExportDrawer] = useState(false);
  const [customPath, setCustomPath] = useState('');
  const [pathLoading, setPathLoading] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);

  const refreshWorkspace = async () => {
    try {
      const [statusRes, historyRes, comparisonRes] = await Promise.all([
        fetch('/api/status'),
        fetch('/api/history'),
        fetch('/api/comparison'),
      ]);
      if (statusRes.ok) setStatus(await statusRes.json());
      if (historyRes.ok) setHistoryPayload(await historyRes.json());
      if (comparisonRes.ok) setComparison(await comparisonRes.json());
    } catch (e) {
      console.log('Workspace polling...', e);
    }
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (e) {
      console.log('Status polling...', e);
    }
  };

  useEffect(() => {
    refreshWorkspace();
    const interval = setInterval(refreshWorkspace, 4000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const revealNodes = document.querySelectorAll('[data-scroll-reveal], .nf-scroll-reveal');
    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        revealObserver.unobserve(entry.target);
      });
    }, { threshold: 0.14, rootMargin: '0px 0px -12% 0px' });

    revealNodes.forEach((node) => revealObserver.observe(node));
    const readyFrame = requestAnimationFrame(() => setScrollReady(true));
    let scrollFrame = null;
    const updateScrollState = () => {
      if (scrollFrame) return;
      scrollFrame = requestAnimationFrame(() => {
        const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
        const progress = Math.min(1, Math.max(0, window.scrollY / maxScroll));
        document.documentElement.style.setProperty('--nf-scroll-y', `${window.scrollY}px`);
        setScrollProgress(progress);
        setNavScrolled(window.scrollY > 28);
        scrollFrame = null;
      });
    };

    updateScrollState();
    window.addEventListener('scroll', updateScrollState, { passive: true });
    window.addEventListener('resize', updateScrollState);
    return () => {
      cancelAnimationFrame(readyFrame);
      if (scrollFrame) cancelAnimationFrame(scrollFrame);
      revealObserver.disconnect();
      window.removeEventListener('scroll', updateScrollState);
      window.removeEventListener('resize', updateScrollState);
    };
  }, [isMutationLoop]);

  useEffect(() => {
    const magneticNodes = [...document.querySelectorAll('.nf-magnetic')];
    const physicsSurfaces = [...document.querySelectorAll('.nf-physics-surface')];
    const states = [
      ...magneticNodes.map((node) => ({ node, type: 'magnetic', current: { x: 0, y: 0 }, target: { x: 0, y: 0 }, velocity: { x: 0, y: 0 } })),
      ...physicsSurfaces.map((node) => ({ node, type: 'surface', current: { x: 0, y: 0 }, target: { x: 0, y: 0 }, velocity: { x: 0, y: 0 } })),
    ];
    let frame = null;

    const settle = () => {
      if (frame) return;
      const tick = () => {
        let moving = false;
        states.forEach((state) => {
          const spring = state.type === 'surface' ? 0.085 : 0.12;
          const damping = state.type === 'surface' ? 0.78 : 0.72;
          state.velocity.x += (state.target.x - state.current.x) * spring;
          state.velocity.y += (state.target.y - state.current.y) * spring;
          state.velocity.x *= damping;
          state.velocity.y *= damping;
          state.current.x += state.velocity.x;
          state.current.y += state.velocity.y;
          if (
            Math.abs(state.current.x - state.target.x) > 0.08
            || Math.abs(state.current.y - state.target.y) > 0.08
            || Math.abs(state.velocity.x) > 0.02
            || Math.abs(state.velocity.y) > 0.02
          ) moving = true;
          if (state.type === 'surface') {
            state.node.style.setProperty('--nf-surface-x', `${state.current.x * 1.5}px`);
            state.node.style.setProperty('--nf-surface-y', `${state.current.y * 1.5}px`);
            state.node.style.setProperty('--nf-surface-rx', `${state.current.y * -0.035}deg`);
            state.node.style.setProperty('--nf-surface-ry', `${state.current.x * 0.035}deg`);
          } else {
            state.node.style.setProperty('--nf-magnetic-x', `${state.current.x}px`);
            state.node.style.setProperty('--nf-magnetic-y', `${state.current.y}px`);
          }
        });
        frame = moving ? requestAnimationFrame(tick) : null;
      };
      frame = requestAnimationFrame(tick);
    };

    const bindings = states.map((state) => {
      const move = (event) => {
        const rect = state.node.getBoundingClientRect();
        const x = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
        const y = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
        if (state.type === 'surface') {
          state.target.x = x * 14;
          state.target.y = y * 14;
        } else {
          state.target.x = x * 6;
          state.target.y = y * 4;
        }
        settle();
      };
      const leave = () => {
        state.target.x = 0;
        state.target.y = 0;
        settle();
      };
      state.node.addEventListener('pointermove', move);
      state.node.addEventListener('pointerleave', leave);
      return { node: state.node, move, leave };
    });

    return () => {
      if (frame) cancelAnimationFrame(frame);
      bindings.forEach(({ node, move, leave }) => {
        node.removeEventListener('pointermove', move);
        node.removeEventListener('pointerleave', leave);
      });
    };
  }, [isMutationLoop, scrollReady]);

  const handleStartMutation = async () => {
    if (mutating) return;
    setMutating(true);
    setSpeech('Initiating genetic mutation sweep: recombining filter kernels, pooling branches, and activation functions...');

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/search`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      socket.send(JSON.stringify({
        generations: 3,
        population: 4,
        proxy_epochs: 2,
        dataset_dir: status?.checkpoint?.metadata?.dataset?.dataset_root,
      }));
    };

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'candidate_evaluated') {
          setSpeech(`Candidate #${msg.data.candidate_id} evaluated: ${msg.data.accuracy?.toFixed(1)}% accuracy (${msg.data.parameters} params).`);
        } else if (msg.type === 'search_completed') {
          setMutating(false);
          setSpeech(`Evolution complete! Top candidate hit ${msg.best_result?.accuracy?.toFixed(1)}% accuracy.`);
          refreshWorkspace();
        }
      } catch (err) {
        console.error(err);
      }
    };

    socket.onclose = () => setMutating(false);
    socket.onerror = () => setMutating(false);
  };

  const handleLoadDatasetPath = async () => {
    if (!customPath.trim()) return;
    setPathLoading(true);
    try {
      const res = await fetch('/api/dataset/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: customPath.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to inspect dataset path');
      setSpeech(`Dataset mounted: ${data.metadata.num_classes} classes detected with ${data.metadata.num_samples} samples.`);
      setShowDatasetDrawer(false);
      fetchStatus();
    } catch (err) {
      alert(`Error loading dataset: ${err.message}`);
    } finally {
      setPathLoading(false);
    }
  };

  const checkpoint = status?.checkpoint || {};
  const metadata = checkpoint.metadata || {};
  const metrics = checkpoint.best_test_metrics || {};
  const history = historyPayload?.history || [];
  const search = historyPayload?.search || {};
  const baseline = historyPayload?.manual_baseline || search.manual_baseline || metadata.manual_baseline || {};
  const bestCandidate = history.find((item) => item.candidate_id === search.best_candidate_id)
    || history.reduce((best, item) => (!best || item.score > best.score ? item : best), null);
  const bestAcc = Number.isFinite(metrics.accuracy) ? metrics.accuracy.toFixed(1) : '—';
  const baselineDelta = Number.isFinite(bestCandidate?.accuracy) && Number.isFinite(baseline?.accuracy)
    ? `${bestCandidate.accuracy - baseline.accuracy >= 0 ? '+' : ''}${(bestCandidate.accuracy - baseline.accuracy).toFixed(1)} pp`
    : '—';
  const latency = Number.isFinite(metrics.inference_latency_per_sample_seconds)
    ? `${(metrics.inference_latency_per_sample_seconds * 1000).toFixed(1)} ms`
    : '—';
  const dataset = metadata.dataset || {};
  const activeModel = status?.active_model;
  const paramCount = metadata.parameters ? (metadata.parameters >= 1e6 ? `${(metadata.parameters / 1e6).toFixed(2)}M` : `${Math.round(metadata.parameters / 1e3)}k`) : '—';
  const datasetLabel = dataset.dataset_root
    ? `${dataset.dataset_root.split('/').filter(Boolean).pop()} (${dataset.num_classes} classes)`
    : 'no dataset';
  const leader = (comparison?.models || []).filter((m) => m.status === 'ready' && m.metrics)
    .reduce((best, m) => (!best || m.metrics.accuracy > best.metrics.accuracy ? m : best), null);
  const generationData = useMemo(() => {
    const byGeneration = new Map();
    history.forEach((item) => {
      const generation = Number(item.generation);
      if (!generation) return;
      const current = byGeneration.get(generation);
      if (!current || Number(item.accuracy || 0) > Number(current.accuracy || 0)) {
        byGeneration.set(generation, item);
      }
    });
    const rows = [...byGeneration.entries()].sort((a, b) => a[0] - b[0]);
    if (!rows.length) return undefined;
    return rows.map(([gen, item]) => ({
      gen,
      acc: Number(item.accuracy || 0),
      f1: Number(item.f1 || 0),
      name: gen === rows.at(-1)[0] ? 'Current best' : (item.mutation_operations?.[0]?.field || 'Architecture shift'),
      desc: item.mutation_operations?.map((op) => `${op.field}${op.from !== undefined ? ` ${op.from} → ${op.to}` : ''}`).join(' · ') || 'Elite preserved or mutation recorded',
      params: `${(Number(item.parameters || 0) / 1000).toFixed(1)}k`,
      champion: item.candidate_id === search.best_candidate_id,
    }));
  }, [history, historyPayload]);
  const selectedGenerationRecord = useMemo(() => {
    const candidates = history.filter((item) => Number(item.generation) === Number(selectedGen));
    return candidates.reduce((best, item) => (!best || Number(item.accuracy || 0) > Number(best.accuracy || 0) ? item : best), null);
  }, [history, selectedGen]);
  const selectedConfig = selectedGenerationRecord?.config || checkpoint.config;

  const jumpTo = (id) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const completeOpening = useCallback(() => {
    setShowOpening(false);
  }, []);

  const toggleTheme = useCallback(() => {
    soundEngine.tick(1500, 0.01);
    setTheme((current) => {
      const next = current === 'light' ? 'dark' : 'light';
      try {
        localStorage.setItem('neuroforge-theme', next);
      } catch {
        // The visual toggle still works when storage is unavailable.
      }
      return next;
    });
  }, []);

  const toggleSound = useCallback(() => {
    const next = soundEngine.toggle();
    setSoundEnabled(next);
  }, []);

  const cfg = checkpoint.config || {};
  const inputSize = cfg.input_size || 64;
  const pyTorchInferenceCode = `# NEUROFORGE champion inference (download the model bundle from the export bench)
import json, torch
from PIL import Image
import torchvision.transforms as T
from neuroforge_core import DynamicCNN   # rebuilds the exact evolved topology

bundle = "path/to/unzipped-bundle"
config = json.load(open(f"{bundle}/config.json"))   # ${cfg.num_layers || '?'} layers, filters ${JSON.stringify(cfg.filters || [])}, ${cfg.block_type || 'plain'} blocks
labels = json.load(open(f"{bundle}/labels.json"))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = DynamicCNN(config, num_classes=len(labels))
model.load_state_dict(torch.load(f"{bundle}/weights.pth", map_location=device))
model.to(device).eval()

prep = T.Compose([T.Resize((${inputSize}, ${inputSize})), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
x = prep(Image.open("sample.jpg").convert("RGB")).unsqueeze(0).to(device)
probs = model(x).softmax(dim=1)[0]
print(labels[int(probs.argmax())], f"{float(probs.max()):.1%}")
`;

  if (isMutationLoop) {
    return (
      <>
        {showOpening && <CinematicSplash onComplete={completeOpening} />}
        <MutationLoopPage
          theme={theme}
          onToggleTheme={toggleTheme}
          scrollReady={scrollReady}
          scrollProgress={scrollProgress}
          navScrolled={navScrolled}
          onReturn={() => { window.location.href = '/'; }}
        />
      </>
    );
  }

  return (
      <div className={`nf-atlas nf-theme-${theme} ${showOpening ? 'nf-is-loading' : 'nf-is-entered'} ${scrollReady ? 'nf-scroll-ready' : ''} min-h-screen selection:bg-[#d5ec42] selection:text-[#111315]`}>
      <div className="nf-scroll-progress" aria-hidden="true"><span style={{ transform: `scaleX(${scrollProgress})` }} /></div>
      {showOpening && <CinematicSplash onComplete={completeOpening} />}
      <div className="nf-noise" aria-hidden="true" />

      <header className={`nf-nav ${navScrolled ? 'nf-nav-scrolled' : ''}`}>
        <a href="#top" className="nf-brand" aria-label="NEUROFORGE home">
          <span className="nf-brand-mark">N</span>
          <span className="nf-brand-name">NEUROFORGE</span>
          <span className="nf-brand-note">model evolution studio</span>
        </a>
        <nav className="nf-nav-links" aria-label="Primary navigation">
          <a href="#evolution">Family tree</a>
          <a href="#evidence">Evidence</a>
          <a href="#coliseum">Coliseum</a>
          <a href="#crucible">Live specimen</a>
          <a href="#sequencer">DNA sequencer</a>
          <a href="#ship">Champion card</a>
        </nav>
        <div className="nf-nav-actions">
          <span className="nf-status-dot" />
          <span>{String(status?.device || 'MPS').toUpperCase()}</span>
          <button 
            className="nf-sound-toggle" 
            onClick={toggleSound} 
            aria-label={`Switch sound ${soundEnabled ? 'off' : 'on'}`}
            title={soundEnabled ? "Tactile Audio Active" : "Tactile Audio Muted"}
          >
            {soundEnabled ? (
              <div className="nf-audio-bars" aria-hidden="true">
                <span className="bar bar-1" />
                <span className="bar bar-2" />
                <span className="bar bar-3" />
              </div>
            ) : (
              <VolumeX size={14} />
            )}
            <span>{soundEnabled ? 'Audio' : 'Muted'}</span>
          </button>
          <button className="nf-theme-toggle" onClick={toggleTheme} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}>
            {theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}
            {theme === 'light' ? 'Dark' : 'Light'}
          </button>
        </div>
      </header>

      <main id="top" className="nf-main">
         <section className="nf-hero nf-section-anchor" data-scroll-reveal="hero">
          <div className="nf-hero-copy">
            <div className="nf-kicker"><span className="nf-kicker-line" /> NEUROFORGE / a living search process</div>
            <h1>Models are<br /><em>forged, not found.</em></h1>
            <p className="nf-hero-lede">
              A model-search instrument that lets architectures evolve in public—then lets you inspect the evidence behind the winner.
            </p>
            <div className="nf-hero-actions">
               <button className="nf-primary-button nf-magnetic" onClick={() => { window.location.href = '/?view=loop'; }}>
                <Play size={15} fill="currentColor" /> Enter the mutation loop
              </button>
               <button className="nf-text-button nf-magnetic" onClick={() => setShowDatasetDrawer(true)}>
                Mount a dataset <ArrowUpRight size={15} />
              </button>
            </div>
            <div className="nf-hero-footnote">
              <span>Build / mutate / ship</span>
              <span>Run {metadata.run_id || '—'}</span>
              <span>{dataset.num_samples || '—'} samples · {dataset.num_classes || '—'} classes</span>
            </div>
          </div>

           <div className="nf-hero-specimen nf-physics-surface">
            <NeuralCoreCanvas
              generation={selectedGen}
              accuracy={bestAcc}
              mutating={mutating}
              isInferring={isInferring}
              activeConfig={selectedConfig}
              numClasses={dataset.num_classes || 5}
              latencyMs={latency}
              lastInference={lastInference}
            />
            <div className="nf-specimen-explainer">
              <span>Topology specimen</span>
              <p>This is the current CNN blueprint made visible: each cluster is a layer, the connecting paths are feature flow, and the green head is the classifying output.</p>
            </div>
          </div>
        </section>

         <section className="nf-metric-ribbon" aria-label="Current run summary" data-scroll-reveal="ribbon">
          <div className="nf-ribbon-label">What the instrument currently knows</div>
          <div className="nf-ribbon-stat"><strong>{bestAcc}%</strong><span>test accuracy</span></div>
          <div className="nf-ribbon-stat"><strong>{metrics.f1 ? `${metrics.f1.toFixed(1)}%` : '—'}</strong><span>macro F1</span></div>
          <div className="nf-ribbon-stat"><strong>{baselineDelta}</strong><span>validation gain vs manual CNN</span></div>
          <div className="nf-ribbon-stat"><strong>{latency}</strong><span>per-sample latency</span></div>
        </section>

         <section className="nf-evidence-note" aria-label="Why the current score is modest" data-scroll-reveal="evidence">
          <div className="nf-evidence-mark">read the number correctly</div>
          <div>
            <strong>Small, difficult specimen.</strong>
            <p>Current run: {dataset.num_samples || '—'} images across {dataset.num_classes || '—'} classes, scored on a sealed test split of {metrics.sample_count || '—'} images (one image ≈ {metrics.sample_count ? (100 / metrics.sample_count).toFixed(1) : '—'} points).{leader && leader.id !== 'dynamic_cnn' ? ` ${leader.label} currently leads at ${leader.metrics.accuracy.toFixed(1)}%: pretrained on 1.2M ImageNet photos, it needs far less data than a CNN evolved from scratch.` : ''}</p>
          </div>
          <div className="nf-evidence-next"><span>Next lever</span><strong>larger dataset · repeated seeds · random-search baseline</strong></div>
        </section>

         <section id="evolution" className="nf-chapter nf-section-anchor" data-scroll-reveal="chapter">
           <div className="nf-chapter-intro nf-scroll-reveal nf-reveal-intro">
            <span className="nf-chapter-number">01</span>
            <div>
              <p className="nf-kicker">The family tree</p>
              <h2>Every strong model<br /><em>leaves a trace.</em></h2>
            </div>
            <p className="nf-chapter-copy">NEUROFORGE does not hide the search behind a spinner. It records parents, mutations, fitness and trade-offs so the architecture can be taught—not merely admired.</p>
          </div>
           <div className="nf-instrument-frame nf-physics-surface nf-scroll-reveal nf-reveal-instrument">
            <KineticTrajectoryGraph
              generations={generationData}
              baselineAccuracy={baseline.accuracy}
              currentGen={selectedGen}
              onSelectGen={(g) => {
                setSelectedGen(g);
                setSpeech(`Generation ${String(g).padStart(2, '0')} selected. The specimen is morphing to its recorded topology.`);
              }}
            />
          </div>
          <div className="nf-scroll-reveal nf-reveal-instrument" style={{ marginTop: 28 }}>
            <LineageBay history={history} winnerId={search.best_candidate_id} manualBaseline={baseline.f1 !== undefined ? baseline : null}
              numClasses={dataset.num_classes} />
          </div>
        </section>

        <section id="evidence" className="nf-chapter nf-section-anchor" data-scroll-reveal="chapter">
          <div className="nf-chapter-intro nf-chapter-intro-reverse nf-scroll-reveal nf-reveal-intro">
            <span className="nf-chapter-number">01·b</span>
            <div>
              <p className="nf-kicker">The evidence</p>
              <h2>A score is only as good<br /><em>as the split behind it.</em></h2>
            </div>
            <p className="nf-chapter-copy">Validation chose the checkpoint; the sealed test split was opened once. The curve shows how the champion learned, the heatmap shows exactly which classes it confuses.</p>
          </div>
          <div className="nf-scroll-reveal nf-reveal-instrument">
            <TrainingEvidenceBay active={activeModel} />
          </div>
        </section>

        <section id="coliseum" className="nf-chapter nf-section-anchor" data-scroll-reveal="chapter">
          <ModelColiseum models={comparison?.models || []} datasetName={datasetLabel} />
        </section>

         <section id="crucible" className="nf-chapter nf-section-anchor" data-scroll-reveal="chapter">
           <div className="nf-chapter-intro nf-chapter-intro-reverse nf-scroll-reveal nf-reveal-intro">
            <span className="nf-chapter-number">02</span>
            <div>
              <p className="nf-kicker">The live specimen</p>
              <h2>Give the champion<br /><em>something real to see.</em></h2>
            </div>
            <p className="nf-chapter-copy">A one-click sample is a small act of theatre with a serious purpose: observe the forward pass, the class distribution and the latency as they happen.</p>
          </div>
           <div className="nf-instrument-frame nf-instrument-frame-warm nf-physics-surface nf-scroll-reveal nf-reveal-instrument">
            <InteractiveCrucible
              onSetSpeech={(s) => setSpeech(s)}
              onInferStart={() => setIsInferring(true)}
              onInferEnd={() => setIsInferring(false)}
              onInferResult={(res) => setLastInference(res)}
            />
          </div>
        </section>

        <section id="sequencer" className="nf-chapter nf-section-anchor" data-scroll-reveal="chapter">
          <ArchitectureSequencer
            initialConfig={selectedConfig}
            numClasses={dataset.num_classes || 5}
            championAcc={search.best_metrics?.accuracy || 0}
            proxyEpochs={search.proxy_epochs || 3}
          />
        </section>

         <section id="ship" className="nf-ship-section nf-physics-surface nf-section-anchor" data-scroll-reveal="ship">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            <div className="lg:col-span-7">
              <p className="nf-kicker">03 / The handoff</p>
              <h2>When the experiment<br /><em>earns a place in the world.</em></h2>
              <p className="nf-chapter-copy">
                Keep the weights, the blueprint and the receipt together. Every forged champion earns an authentic SSR Holographic Trading Card with verified Apple Silicon tensor benchmarks.
              </p>
              <div className="nf-ship-actions mt-6">
                <button className="nf-primary-button nf-primary-button-dark nf-magnetic" onClick={() => setShowExportDrawer(true)}>
                  <Download size={15} /> Open export bench
                </button>
                <button className="nf-text-button nf-text-button-dark nf-magnetic" onClick={() => setShowDatasetDrawer(true)}>
                  Change specimen <FolderOpen size={15} />
                </button>
              </div>
            </div>

            <div className="lg:col-span-5 flex justify-center">
              <HolographicSpecimenCard
                modelName={checkpoint?.metadata?.run_id ? `NEURO-TITAN // ${checkpoint.metadata.run_id}` : 'NEURO-TITAN // G5-CHAMPION'}
                accuracy={metrics.accuracy ? metrics.accuracy.toFixed(1) : bestAcc}
                f1={metrics.f1 ? metrics.f1.toFixed(1) : '—'}
                latency={metrics.inference_latency_per_sample_seconds ? (metrics.inference_latency_per_sample_seconds * 1000).toFixed(2) : '—'}
                paramCount={paramCount}
                datasetName={datasetLabel}
                numClasses={dataset.num_classes || 5}
                generation={selectedGen}
                config={selectedConfig}
              />
            </div>
          </div>
        </section>
      </main>

      {showDatasetDrawer && (
        <div className="nf-drawer-backdrop" onClick={() => setShowDatasetDrawer(false)}>
          <aside className="nf-drawer" onClick={(event) => event.stopPropagation()}>
            <div>
              <div className="nf-drawer-heading"><div><p className="nf-kicker">Chamber 01</p><h3>Dataset intake</h3></div><button className="nf-close" onClick={() => setShowDatasetDrawer(false)}><X size={18} /></button></div>
              <div className="nf-drawer-specimen">
                <span>Active specimen</span>
                <strong>{dataset.modality === 'audio' ? 'Audio classification' : 'Crop pest classification'}</strong>
                <p>{dataset.num_samples || '—'} samples · {dataset.num_classes || '—'} classes</p>
                <small>{dataset.class_names?.join(' · ') || 'No dataset mounted'}</small>
              </div>
              <label className="nf-field-label">Mount a local directory</label>
              <div className="nf-field-row"><input value={customPath} onChange={(e) => setCustomPath(e.target.value)} placeholder="/path/to/dataset" /><button className="nf-small-button" onClick={handleLoadDatasetPath} disabled={pathLoading}>{pathLoading ? 'Reading…' : 'Mount'}</button></div>
              <p className="nf-drawer-note">The path is read by the local NEUROFORGE engine. Nothing leaves this machine.</p>
            </div>
            <div className="nf-drawer-footer">A dataset is not just input. It defines the search universe.</div>
          </aside>
        </div>
      )}

      {showExportDrawer && (
        <div className="nf-drawer-backdrop" onClick={() => setShowExportDrawer(false)}>
          <aside className="nf-drawer nf-drawer-wide" onClick={(event) => event.stopPropagation()}>
            <div>
              <div className="nf-drawer-heading"><div><p className="nf-kicker">Chamber 03</p><h3>Ship the evidence</h3></div><button className="nf-close" onClick={() => setShowExportDrawer(false)}><X size={18} /></button></div>
              <div className="flex justify-center mb-6">
                <HolographicSpecimenCard
                  modelName={checkpoint?.metadata?.run_id ? `NEURO-TITAN // ${checkpoint.metadata.run_id}` : 'NEURO-TITAN // G5-CHAMPION'}
                  accuracy={metrics.accuracy ? metrics.accuracy.toFixed(1) : bestAcc}
                  f1={metrics.f1 ? metrics.f1.toFixed(1) : '—'}
                  latency={metrics.inference_latency_per_sample_seconds ? (metrics.inference_latency_per_sample_seconds * 1000).toFixed(2) : '—'}
                  paramCount={paramCount}
                  datasetName={datasetLabel}
                  numClasses={dataset.num_classes || 5}
                  generation={selectedGen}
                  config={selectedConfig}
                />
              </div>
              <div className="nf-export-grid">
                <a href="/api/download/weights" download><Download size={18} /><strong>Weights</strong><span>best_model_weights.pth</span></a>
                <a href="/api/download/config" download><Layers size={18} /><strong>Blueprint</strong><span>best_model_config.json</span></a>
                {activeModel?.bundle_id && <a href="/api/download/bundle" download><Award size={18} /><strong>Full bundle</strong><span>weights · labels · splits · metrics · lineage</span></a>}
              </div>
              <div className="nf-code-heading"><span>Production inference snippet</span><button onClick={() => { navigator.clipboard.writeText(pyTorchInferenceCode); setCopiedCode(true); setTimeout(() => setCopiedCode(false), 2000); }}>{copiedCode ? <Check size={14} /> : <Copy size={14} />} {copiedCode ? 'Copied' : 'Copy code'}</button></div>
              <pre className="nf-code-block">{pyTorchInferenceCode}</pre>
            </div>
            <div className="nf-drawer-footer">Configuration · weights · preprocessing · provenance</div>
          </aside>
        </div>
      )}

      <footer className="nf-footer"><span>NEUROFORGE / architecture search, made legible</span><span>Build · mutate · ship</span></footer>
    </div>
  );
}
