import { LineageBay, TrainingEvidenceBay } from '../evidence/EvidencePanels';
import ArchitectureSequencer from './ArchitectureSequencer';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, BarChart3, Database, Download, Info, Moon, Play, RotateCcw, Save, Sun, Trash2, UploadCloud, Zap, Award, Check, Layers } from 'lucide-react';
import soundEngine from '../utils/soundEngine';

const initialSearch = { generations: 3, population: 4, proxyEpochs: 2 };
const SAVED_RUNS_KEY = 'neuroforge:saved-runs:v1';

const readSavedRuns = () => {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(SAVED_RUNS_KEY) || '[]');
    return Array.isArray(parsed) ? parsed.filter((run) => run && run.id && Number.isFinite(Number(run.accuracy))) : [];
  } catch {
    return [];
  }
};

const mutationLabel = (field = '') => {
  const indexed = field.match(/^(filters|kernel_sizes)\[(\d+)\]$/);
  if (indexed?.[1] === 'filters') return `filters in block ${Number(indexed[2]) + 1}`;
  if (indexed?.[1] === 'kernel_sizes') return `kernel size in block ${Number(indexed[2]) + 1}`;
  return ({
    activation: 'activation',
    batch_norm: 'batch normalization',
    dropout: 'dropout',
    num_layers: 'layer count',
    pooling: 'pooling',
  })[field] || field.replaceAll('_', ' ');
};

const mutationValue = (value, field = '') => {
  if (typeof value === 'boolean') return value ? 'on' : 'off';
  if (field === 'dropout') return `${Math.round(Number(value) * 100)}%`;
  const indexed = field.match(/^kernel_sizes\[(\d+)\]$/);
  if (indexed) return `${value}×${value}`;
  return String(value ?? '—');
};

const describeMutations = (candidate) => {
  const operations = candidate?.mutation_operations || [];
  if (!operations.length) return Number(candidate?.generation) === 1
    ? 'Seed architecture; no mutation applied.'
    : 'No mutation recorded for this candidate.';
  return operations.map(({ field, from, to }) => {
    if (field === 'elite') return 'elite: best design carried forward unchanged';
    // On/off toggles (BatchNorm, pooling) record only the new state.
    if (from === undefined) return `${mutationLabel(field)} switched ${mutationValue(to, field)}`;
    return `${mutationLabel(field)}: ${mutationValue(from, field)} → ${mutationValue(to, field)}`;
  }).join(' · ');
};

export default function MutationLoopPage({ theme, onToggleTheme, onReturn, scrollReady, scrollProgress, navScrolled }) {
  const [status, setStatus] = useState(null);
  const [historyPayload, setHistoryPayload] = useState({ history: [], metadata: {} });
  const [path, setPath] = useState('');
  const [busyDataset, setBusyDataset] = useState(false);
  const [datasetError, setDatasetError] = useState('');
  const [search, setSearch] = useState(initialSearch);
  const [searching, setSearching] = useState(false);
  const [searchStage, setSearchStage] = useState('ready');
  const socketRef = useRef(null);
  const [currentGeneration, setCurrentGeneration] = useState(1);
  const [liveCandidates, setLiveCandidates] = useState([]);
  const [trainingEpochs, setTrainingEpochs] = useState(20);
  const [training, setTraining] = useState(false);
  const [trainingMessage, setTrainingMessage] = useState('');
  const [comparison, setComparison] = useState(null);
  const [comparisonError, setComparisonError] = useState('');
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [resnetEpochs, setResnetEpochs] = useState(20);
  const [resnetTraining, setResnetTraining] = useState(false);
  const [selectedTransferModel, setSelectedTransferModel] = useState('resnet18');
  const [savedRuns, setSavedRuns] = useState(readSavedRuns);
  const [jobProgress, setJobProgress] = useState([]);

  // While a training job runs on the server, stream its per-epoch progress.
  useEffect(() => {
    if (!training && !resnetTraining) return undefined;
    const id = setInterval(async () => {
      try {
        const response = await fetch('/api/job');
        if (response.ok) setJobProgress(((await response.json()).progress || []).filter((p) => p.epoch));
      } catch { /* keep the last snapshot */ }
    }, 1500);
    return () => clearInterval(id);
  }, [training, resnetTraining]);
  const liveCurve = jobProgress.length
    ? jobProgress.map((p) => ({ epoch: p.epoch, train_accuracy: p.train_acc, valid_f1: p.val_f1, valid_accuracy: p.val_acc }))
    : null;
  const liveBest = liveCurve ? liveCurve.reduce((a, b) => (b.valid_f1 > a.valid_f1 ? b : a)) : null;
  const uploadRef = useRef(null);

  const refresh = async () => {
    try {
      setComparisonLoading(true);
      const [statusResponse, historyResponse, comparisonResponse] = await Promise.all([
        fetch('/api/status'),
        fetch('/api/history'),
        fetch('/api/comparison'),
      ]);
      if (statusResponse.ok) setStatus(await statusResponse.json());
      if (historyResponse.ok) setHistoryPayload(await historyResponse.json());
      if (comparisonResponse.ok) {
        setComparison(await comparisonResponse.json());
        setComparisonError('');
      } else {
        const payload = await comparisonResponse.json().catch(() => ({}));
        setComparisonError(payload.detail || 'Comparison data is unavailable.');
      }
    } catch (error) {
      setDatasetError(`Workspace unavailable: ${error.message}`);
    } finally {
      setComparisonLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  // Deep links from the home page (e.g. #nf-loop-evidence) land after the data renders.
  useEffect(() => {
    if (!window.location.hash) return undefined;
    const timer = setTimeout(() => document.querySelector(window.location.hash)?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 1400);
    return () => clearTimeout(timer);
  }, []);

  const workspace = status?.workspace || {};
  const dataset = workspace.dataset_metadata || historyPayload.metadata?.dataset || null;
  const history = liveCandidates.length ? liveCandidates : (historyPayload.history || []);
  const generations = useMemo(() => {
    const byGeneration = new Map();
    history.forEach((candidate) => {
      const generation = Number(candidate.generation);
      if (!generation) return;
      const current = byGeneration.get(generation);
      if (!current || Number(candidate.score || 0) > Number(current.score || 0)) {
        byGeneration.set(generation, candidate);
      }
    });
    return [...byGeneration.entries()].sort((a, b) => a[0] - b[0]);
  }, [history]);
  const latest = historyPayload.metadata?.held_out_test_metrics || status?.checkpoint?.best_test_metrics || {};
  const comparisonModels = comparison?.models || [];
  const trainingComplete = Boolean(historyPayload.metadata?.training_complete || status?.checkpoint?.metadata?.training_complete);
  const activePhase = training || resnetTraining
    ? 2
    : searching
      ? 1
      : trainingComplete
        ? 2
        : history.length
        ? 1
        : dataset
          ? 0
          : 0;
  const comparisonMetricKeys = ['accuracy', 'precision', 'recall', 'f1', 'mse', 'rmse'];
  const bestByMetric = useMemo(() => {
    const best = {};
    comparisonMetricKeys.forEach((key) => {
      const lowerIsBetter = key === 'mse' || key === 'rmse';
      let winner = null;
      (comparison?.models || []).forEach((m) => {
        const v = m.metrics?.[key];
        if (v === null || v === undefined) return;
        if (!winner || (lowerIsBetter ? v < winner.v : v > winner.v)) winner = { id: m.id, v };
      });
      best[key] = winner?.id;
    });
    return best;
  }, [comparison]);
  const activeDatasetRoot = dataset?.dataset_root || workspace.dataset_dir || '';
  const historyDatasetRoot = historyPayload.search?.dataset?.dataset_root || '';
  const hasSearchResults = Boolean(dataset && history.length && (liveCandidates.length || !historyDatasetRoot || historyDatasetRoot === activeDatasetRoot));
  const checkpointDatasetRoot = status?.checkpoint?.metadata?.dataset?.dataset_root || historyPayload.metadata?.dataset?.dataset_root || '';
  const trainedForActiveDataset = Boolean(dataset && trainingComplete && (!checkpointDatasetRoot || checkpointDatasetRoot === activeDatasetRoot));
  const champion = useMemo(() => {
    if (!history.length) return null;
    const bestId = historyPayload.search?.best_candidate_id || historyPayload.metadata?.best_candidate_id;
    return history.find((candidate) => candidate.candidate_id === bestId)
      || history.reduce((best, candidate) => (!best || Number(candidate.score || 0) > Number(best.score || 0) ? candidate : best), null);
  }, [history, historyPayload]);
  const championId = champion?.candidate_id;
  const savedSearch = historyPayload.search || {};
  const activeCandidateId = status?.active_model?.candidate_id;
  const activeIsChampion = Boolean(trainedForActiveDataset && championId && activeCandidateId === championId);
  const deviceLabel = String(status?.device || 'CPU').toUpperCase();
  const baselineBusy = training || resnetTraining || searching;

  // Live runs count against the form's budget; a finished run against the budget it actually used.
  const plannedGenerations = searching ? Number(search.generations) || 1 : Number(savedSearch.generations) || Number(search.generations) || 1;
  const plannedPopulation = searching ? Number(search.population) || 2 : Number(savedSearch.population_size) || Number(search.population) || 2;
  const totalExpected = Math.max(1, plannedGenerations * plannedPopulation);
  const evaluatedCount = history.length;
  const searchPercent = Math.min(100, Math.round((evaluatedCount / totalExpected) * 100));

  const currentLeader = useMemo(() => {
    if (!history.length) return null;
    return history.reduce((best, c) => (!best || Number(c.score || 0) > Number(best.score || 0) ? c : best), null);
  }, [history]);

  const [candidateFilter, setCandidateFilter] = useState('all');

  const formatParamCount = (num) => {
    const n = Number(num);
    if (!Number.isFinite(n)) return '—';
    if (n >= 1000000) return `${(n / 1000000).toFixed(2)}M params`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}k params`;
    return `${n} params`;
  };

  const formatCandidateDisplayTitle = (id, index, gen) => {
    if (!id) return `Candidate #${index + 1}`;
    const match = id.match(/G(\d+)-C(\d+)/i);
    if (match) {
      return `Candidate ${match[2]} · Gen ${Number(match[1])}`;
    }
    return `Candidate #${index + 1} (Gen ${gen || 1})`;
  };

  const bestCandidateAccuracy = useMemo(() => {
    if (!history.length) return 0;
    return Math.max(...history.map((c) => Number(c.accuracy || 0)));
  }, [history]);

  const lastGeneration = history.reduce((max, c) => Math.max(max, Number(c.generation) || 0), 0);
  const filteredCandidates = useMemo(() => {
    if (candidateFilter === 'top') {
      return [...history].sort((a, b) => Number(b.score || 0) - Number(a.score || 0)).slice(0, 5);
    }
    if (candidateFilter === 'latest') {
      return history.filter((c) => Number(c.generation) === lastGeneration);
    }
    return history;
  }, [history, candidateFilter, lastGeneration]);

  const readResponse = async (response) => {
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Request failed');
    return payload;
  };

  const mountPath = async () => {
    if (!path.trim()) return;
    setBusyDataset(true);
    setDatasetError('');
    try {
      await readResponse(await fetch('/api/dataset/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: path.trim() }),
      }));
      await refresh();
    } catch (error) {
      setDatasetError(error.message);
    } finally {
      setBusyDataset(false);
    }
  };

  const uploadDataset = async (file) => {
    if (!file) return;
    setBusyDataset(true);
    setDatasetError('');
    try {
      await readResponse(await fetch('/api/dataset/upload-zip', {
        method: 'POST',
        body: (() => { const form = new FormData(); form.append('file', file); return form; })(),
      }));
      await refresh();
    } catch (error) {
      setDatasetError(error.message);
    } finally {
      setBusyDataset(false);
      if (uploadRef.current) uploadRef.current.value = '';
    }
  };

  const removeDataset = async () => {
    if (!dataset) return;
    const shouldDelete = Boolean(workspace.dataset_owned);
    if (!window.confirm(shouldDelete ? 'Delete this uploaded dataset extraction from the local workspace?' : 'Unmount this local dataset path?')) return;
    setBusyDataset(true);
    setDatasetError('');
    try {
      await readResponse(await fetch('/api/dataset/unload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delete_uploaded: shouldDelete }),
      }));
      setLiveCandidates([]);
      await refresh();
    } catch (error) {
      setDatasetError(error.message);
    } finally {
      setBusyDataset(false);
    }
  };

  const startSearch = () => {
    if (searching || !dataset?.dataset_root) return;
    setSearching(true);
    setSearchStage('connecting');
    setLiveCandidates([]);
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/search`);
    socketRef.current = socket;
    socket.onopen = () => {
      socket.send(JSON.stringify({
        generations: Number(search.generations),
        population: Number(search.population),
        proxy_epochs: Number(search.proxyEpochs),
        dataset_dir: dataset.dataset_root,
      }));
    };
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'started') setSearchStage('evaluating');
      if (message.type === 'candidate_evaluated') {
        setSearchStage('evaluating');
        setCurrentGeneration(Number(message.data.generation || 1));
        setLiveCandidates((previous) => [message.data, ...previous]);
        soundEngine.synapsePulse(Number(message.data.generation || 1), Number(search.generations || 4));
      }
      if (message.type === 'generation_completed') {
        setCurrentGeneration(Number(message.generation));
        setSearchStage(`generation-${message.generation}`);
        soundEngine.championChord(false);
      }
      if (message.type === 'search_completed') {
        setSearchStage(message.stopped_early ? 'stopped' : 'complete');
        setSearching(false);
        soundEngine.championChord(true);
        refresh();
      }
      if (message.type === 'error') {
        setSearchStage(message.message);
        setSearching(false);
      }
    };
    socket.onerror = () => {
      setSearchStage('Connection interrupted');
      setSearching(false);
    };
    socket.onclose = () => setSearching(false);
  };

  const stopSearch = () => {
    socketRef.current?.send(JSON.stringify({ type: 'stop' }));
    setSearchStage('stopping');
  };

  const stageLabel = (stage) => {
    if (stage === 'ready') return 'ready';
    if (stage === 'connecting') return 'connecting…';
    if (stage === 'evaluating') return `evaluating on ${deviceLabel}`;
    if (stage === 'stopping') return 'stopping after this generation…';
    if (stage === 'stopped') return 'stopped, best design kept';
    if (stage === 'complete') return 'search complete';
    const done = /^generation-(\d+)$/.exec(stage);
    return done ? `generation ${done[1]} done` : stage;
  };

  const trainChampion = async (candidateId = null, epochsOverride = null) => {
    if (training || !dataset) return;
    const epochs = Number(epochsOverride || trainingEpochs);
    setTraining(true);
    setJobProgress([]);
    setTrainingMessage(`Training ${candidateId ? candidateId.split('-').slice(-2).join(' ') : 'the selected architecture'} for up to ${epochs} epochs…`);
    try {
      const result = await readResponse(await fetch('/api/train/full', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs, candidate_id: candidateId }),
      }));
      const testAccuracy = result.metrics?.held_out_test_metrics?.accuracy;
      setTrainingMessage(`Checkpoint saved (best epoch ${result.metrics?.best_epoch}). Test accuracy on the sealed split: ${testAccuracy?.toFixed?.(1) || '—'}%.`);
      await refresh();
    } catch (error) {
      setTrainingMessage(`Training failed: ${error.message}`);
    } finally {
      setTraining(false);
    }
  };

  const transferOptions = [
    { id: 'manual_cnn', label: 'Manual CNN', fullName: 'Manual 2-layer CNN (same protocol)' },
    { id: 'resnet18', label: 'ResNet-18', fullName: 'ResNet-18 (Deep Residual)' },
    { id: 'mobilenet_v3', label: 'MobileNet-V3', fullName: 'MobileNet-V3 Small (Google Edge)' },
    { id: 'shufflenet_v2', label: 'ShuffleNet-V2', fullName: 'ShuffleNet-V2 0.5× (Compact)' },
    { id: 'efficientnet_b0', label: 'EfficientNet-B0', fullName: 'EfficientNet-B0 (Compound Scaled)' },
  ];

  const runTransferComparison = async () => {
    if (resnetTraining || !dataset) return;
    setResnetTraining(true);
    setComparisonError('');
    try {
      const result = await readResponse(await fetch('/api/train/transfer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          model_type: selectedTransferModel, 
          epochs: Number(resnetEpochs) 
        }),
      }));
      setComparison(result.comparison);
    } catch (error) {
      setComparisonError(error.message);
    } finally {
      setResnetTraining(false);
    }
  };

  const scrollToPhase = (phase) => {
    document.getElementById(`nf-loop-phase-${phase}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const formatMetric = (value, key) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return key === 'mse' || key === 'rmse' ? Number(value).toFixed(4) : `${Number(value).toFixed(1)}%`;
  };

  const saveCurrentRun = () => {
    if (!dataset || !champion) return;
    const testMetrics = historyPayload.metadata?.held_out_test_metrics || status?.checkpoint?.best_test_metrics || {};
    const hasHeldOutScore = trainedForActiveDataset && Number.isFinite(Number(testMetrics.accuracy));
    const accuracy = hasHeldOutScore ? Number(testMetrics.accuracy) : Number(champion.accuracy);
    if (!Number.isFinite(accuracy)) return;
    const latencySeconds = hasHeldOutScore ? Number(testMetrics.inference_latency_per_sample_seconds) : NaN;
    const record = {
      id: `${Date.now()}-${champion.candidate_id || 'champion'}`,
      createdAt: new Date().toISOString(),
      candidateId: champion.candidate_id || 'current champion',
      generation: Number(champion.generation) || null,
      scoreBasis: hasHeldOutScore ? 'held-out test' : 'proxy score',
      accuracy,
      f1: hasHeldOutScore && Number.isFinite(Number(testMetrics.f1)) ? Number(testMetrics.f1) : Number(champion.f1) || null,
      parameters: Number.isFinite(Number(champion.parameters)) ? Number(champion.parameters) : null,
      latencyMs: Number.isFinite(latencySeconds) && latencySeconds > 0 ? latencySeconds * 1000 : null,
      datasetLabel: `${dataset.modality === 'audio' ? 'Audio' : 'Image'} · ${dataset.num_samples || '—'} samples · ${dataset.num_classes || '—'} classes`,
    };
    const next = [record, ...savedRuns].slice(0, 12);
    setSavedRuns(next);
    try {
      window.localStorage.setItem(SAVED_RUNS_KEY, JSON.stringify(next));
    } catch {
      setDatasetError('This browser could not save the run. Check its local storage settings.');
    }
  };

  const removeSavedRun = (id) => {
    const next = savedRuns.filter((run) => run.id !== id);
    setSavedRuns(next);
    try {
      window.localStorage.setItem(SAVED_RUNS_KEY, JSON.stringify(next));
    } catch {
      setDatasetError('This browser could not update saved runs. Check its local storage settings.');
    }
  };

  return (
    <div className={`nf-atlas nf-loop-page min-h-screen nf-theme-${theme} ${scrollReady ? 'nf-scroll-ready' : ''}`}>
      <div className="nf-scroll-progress" aria-hidden="true"><span style={{ transform: `scaleX(${scrollProgress || 0})` }} /></div>
      <header className={`nf-nav nf-loop-nav ${navScrolled ? 'nf-nav-scrolled' : ''}`}>
        <button className="nf-loop-back nf-magnetic" onClick={onReturn}><ArrowLeft size={16} /> Studio</button>
        <div className="nf-brand"><span className="nf-brand-mark">N</span><span className="nf-brand-name">NEUROFORGE</span><span className="nf-brand-note">mutation loop</span></div>
        <button className="nf-theme-toggle nf-magnetic" onClick={onToggleTheme} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}>
          {theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}
          {theme === 'light' ? 'Dark' : 'Light'} mode
        </button>
      </header>

      <main className="nf-loop-main">
        <section className="nf-loop-heading nf-scroll-reveal" data-scroll-reveal="loop-heading">
          <div>
            <p className="nf-kicker"><span className="nf-kicker-line" /> chamber 01 → 03 / the working instrument</p>
            <h1>Enter the<br /><em>mutation loop.</em></h1>
            <p>Mount the specimen, decide how much to explore, watch generations compete, then forge the selected architecture for a declared number of epochs.</p>
          </div>
          <div className="nf-loop-status"><span className="nf-status-dot" /> {searching ? 'search in progress' : 'instrument ready'}<strong>{dataset ? `${dataset.num_samples} samples` : 'no dataset mounted'}</strong></div>
        </section>

        <section className="nf-loop-phase-rail nf-scroll-reveal" aria-label="Mutation loop phases" data-scroll-reveal="loop-rail">
          {['Mount data', 'Evolve designs', 'Train and ship'].map((phase, index) => (
            <div className={`nf-phase-magnetic nf-magnetic ${index === activePhase ? 'is-active' : ''}`} key={phase} role="button" tabIndex="0" onClick={() => scrollToPhase(index === 0 ? 'build' : index === 1 ? 'mutate' : 'ship')} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') scrollToPhase(index === 0 ? 'build' : index === 1 ? 'mutate' : 'ship'); }}><span>0{index + 1}</span><strong>{phase}</strong><small>{index === 0 ? 'define the search universe' : index === 1 ? 'breed and rank candidates' : 'turn the winner into weights'}</small></div>
          ))}
        </section>

        <section className="nf-first-run" aria-labelledby="nf-first-run-title">
          <div className="nf-first-run-lead">
            <div className="nf-first-run-meta"><p className="nf-kicker">A first run, guided</p><span>{Number(Boolean(dataset)) + Number(hasSearchResults) + Number(trainedForActiveDataset)} / 3 complete</span></div>
            <h2 id="nf-first-run-title">{!dataset ? 'Start with your data.' : !hasSearchResults ? 'Find a champion.' : !trainedForActiveDataset ? 'Train the strongest candidate.' : 'Put the model to work.'}</h2>
            <p>{!dataset ? 'Mount a labelled image or WAV dataset. The steps below follow this workspace, so the page never invents a demo result.' : !hasSearchResults ? 'Run a proxy search to see candidates evolve. Each result now explains the architectural change behind its score.' : !trainedForActiveDataset ? 'Search scores are only a shortlist. Train the selected architecture before treating its held-out score as evidence.' : 'The champion is trained for this dataset. Try a sample in the live inference chamber.'}</p>
            <button className="nf-loop-button nf-loop-button-accent" onClick={() => {
              if (!dataset) scrollToPhase('build');
              else if (!hasSearchResults) scrollToPhase('mutate');
              else if (!trainedForActiveDataset) scrollToPhase('ship');
              else window.location.href = '/#crucible';
            }}>
              {!dataset ? 'Mount a dataset' : !hasSearchResults ? 'Start a search' : !trainedForActiveDataset ? 'Train the champion' : 'Test a sample'}
            </button>
          </div>
          <ol className="nf-first-run-steps">
            <li className={dataset ? 'is-complete' : 'is-next'}><span>{dataset ? '✓' : '01'}</span><div><strong>Mount data</strong><small>ImageFolder, Roboflow, or supported audio data</small></div></li>
            <li className={hasSearchResults ? 'is-complete' : dataset ? 'is-next' : ''}><span>{hasSearchResults ? '✓' : '02'}</span><div><strong>Evolve candidates</strong><small>Compare recorded mutations and proxy scores</small></div></li>
            <li className={trainedForActiveDataset ? 'is-complete' : hasSearchResults ? 'is-next' : ''}><span>{trainedForActiveDataset ? '✓' : '03'}</span><div><strong>Train, then test</strong><small>Evaluate the champion before inference</small></div></li>
          </ol>
        </section>

        <section id="nf-loop-phase-build" className="nf-loop-grid nf-loop-intake nf-scroll-reveal" data-scroll-reveal="loop-build">
          <div className="nf-loop-panel nf-loop-panel-wide">
            <div className="nf-loop-panel-heading"><div><p className="nf-kicker">01 / build</p><h2>Mount the dataset.</h2></div><Database size={20} /></div>
            <p className="nf-loop-muted">Use a local ImageFolder path or upload a ZIP. Removing a local path only unmounts it; uploaded extractions can be deleted from NEUROFORGE’s workspace.</p>
            <div className="nf-loop-input-row"><input value={path} onChange={(event) => setPath(event.target.value)} placeholder="/path/to/image-folder" /><button className="nf-loop-button" onClick={mountPath} disabled={busyDataset || baselineBusy}>{busyDataset ? 'Reading…' : 'Mount path'}</button></div>
            <div className="nf-loop-upload" onClick={() => { if (!baselineBusy) uploadRef.current?.click(); }} aria-disabled={baselineBusy}><input ref={uploadRef} type="file" accept=".zip" onChange={(event) => uploadDataset(event.target.files?.[0])} /><UploadCloud size={20} /><strong>Upload dataset ZIP</strong><span>ImageFolder or Roboflow single-label export</span></div>
            {baselineBusy && <p className="nf-loop-muted" style={{ marginTop: 10 }}>Dataset controls unlock when the running {searching ? 'search' : 'training'} finishes.</p>}
            {datasetError && <div className="nf-loop-error">{datasetError}</div>}
          </div>
          <div className="nf-loop-panel nf-loop-spec-card">
            <div className="nf-loop-panel-heading"><div><p className="nf-kicker">active specimen</p><h2>{dataset ? 'Dataset mounted' : 'Awaiting data'}</h2></div><span className="nf-loop-live-dot" /></div>
            {dataset ? <><strong className="nf-loop-dataset-name">{dataset.modality === 'audio' ? 'Audio classification' : 'Image classification'}</strong><div className="nf-loop-stat-list"><span>samples <b>{dataset.num_samples}</b></span><span>classes <b>{dataset.num_classes}</b></span><span>format <b>{dataset.format}</b></span></div><div className="nf-loop-tags">{dataset.class_names?.map((name) => <span key={name}>{name}</span>)}</div><button className="nf-loop-remove" onClick={removeDataset} disabled={busyDataset || baselineBusy}><Trash2 size={14} /> {workspace.dataset_owned ? 'Delete uploaded extraction' : 'Unmount local path'}</button></> : <div className="nf-loop-empty">Nothing can mutate until a labelled dataset is mounted.</div>}
          </div>
        </section>

        <section id="nf-loop-phase-mutate" className="nf-loop-panel nf-loop-search-panel nf-scroll-reveal" data-scroll-reveal="loop-mutate">
          <div className="nf-loop-panel-heading"><div><p className="nf-kicker">02 / mutate</p><h2>Set the search budget.</h2></div><span className="nf-loop-search-state">{searching ? <><RotateCcw className="nf-spin" size={14} /> {stageLabel(searchStage)}</> : stageLabel(searchStage)}</span></div>
          <div className="nf-loop-controls">
            <label>Generations<input type="number" min="1" max="20" value={search.generations} disabled={searching} onChange={(event) => setSearch({ ...search, generations: event.target.value })} /></label>
            <label>Population / generation<input type="number" min="2" max="24" value={search.population} disabled={searching} onChange={(event) => setSearch({ ...search, population: event.target.value })} /></label>
            <label>Proxy epochs / candidate<input type="number" min="1" max="5" value={search.proxyEpochs} disabled={searching} onChange={(event) => setSearch({ ...search, proxyEpochs: event.target.value })} /></label>
            {searching ? (
              <button className="nf-loop-button" onClick={stopSearch} disabled={searchStage === 'stopping'}>{searchStage === 'stopping' ? 'Stopping…' : 'Stop search'}</button>
            ) : (
              <button className="nf-loop-button nf-loop-button-accent" onClick={startSearch} disabled={!dataset || training || resnetTraining}><Play size={15} fill="currentColor" /> Start mutation search</button>
            )}
          </div>

          {/* Real-time Evolution Progress Ribbon */}
          {(searching || evaluatedCount > 0) && (
            <div className="nf-search-progress-wrap">
              <div className="nf-search-progress-meta">
                <div className="nf-search-progress-label">
                  <span className={`nf-status-dot ${searching ? 'animate-pulse' : ''}`} style={{ background: searching ? '#00f0ff' : '#00f5a0' }} />
                  <strong>{searching ? `Forging generation ${currentGeneration} of ${search.generations}` : savedSearch.stopped_early ? 'Search stopped early' : 'Search complete'}</strong>
                  <span>· {evaluatedCount} of {totalExpected} candidates evaluated</span>
                </div>
                <strong className="nf-search-progress-pct">{searchPercent}%</strong>
              </div>
              <div className="nf-search-progress-bar">
                <div className="nf-search-progress-fill" style={{ width: `${searchPercent}%` }} />
              </div>
            </div>
          )}

          {/* Current Leader Banner */}
          {currentLeader && (
            <div className="nf-leader-pill">
              <div className="nf-leader-pill-left">
                <span className="nf-crown">👑</span>
                <strong>{searching ? 'Current leader' : 'Search winner'}: {formatCandidateDisplayTitle(currentLeader.candidate_id, 0, currentLeader.generation)}</strong>
                <span className="nf-leader-score">{Number(currentLeader.accuracy).toFixed(1)}% val acc · {Number(currentLeader.f1).toFixed(1)}% F1</span>
                <span className="nf-leader-meta">
                  {currentLeader.parameters ? `· ${(Number(currentLeader.parameters) / 1000).toFixed(1)}k params` : ''} · {describeMutations(currentLeader)}
                </span>
              </div>
              <span className="nf-leader-pill-tag">TOP FITNESS {Number(currentLeader.score).toFixed(1)}</span>
            </div>
          )}

          {/* Dynamic Generation Strip */}
          <div 
            className="nf-generation-strip"
            style={{
              gridTemplateColumns: `repeat(${Math.max(1, Math.min(6, plannedGenerations))}, minmax(0, 1fr))`,
            }}
          >
            {Array.from({ length: Math.max(1, plannedGenerations) }, (_, index) => {
              const gen = index + 1;
              const record = generations.find(([number]) => number === gen)?.[1];
              const isCurrent = searching && currentGeneration === gen;
              const isCompleted = record && (!searching || currentGeneration > gen);
              return (
                <div className={`nf-gen-card ${isCurrent ? 'is-current' : ''} ${isCompleted ? 'is-completed' : ''}`} key={gen}>
                  <div className="nf-gen-card-header">
                    <span>GEN {String(gen).padStart(2, '0')}</span>
                    {isCurrent && <span className="nf-gen-live-tag">● ACTIVE</span>}
                    {isCompleted && <span className="nf-gen-done-tag">✓</span>}
                  </div>
                  <strong>{record ? `${Number(record.accuracy).toFixed(1)}%` : '—'}</strong>
                  <small>
                    {isCurrent 
                      ? `breeding population on ${deviceLabel}…`
                      : record ? `best: ${describeMutations(record)}` : (searching ? 'waiting for candidates' : 'not run')}
                  </small>
                </div>
              );
            })}
          </div>
        </section>

        <section id="nf-loop-phase-ship" className="nf-loop-lower-grid nf-scroll-reveal" data-scroll-reveal="loop-ship">
          <div className="nf-loop-panel nf-loop-feed">
            <div className="nf-loop-panel-heading">
              <div>
                <p className="nf-kicker">live lineage</p>
                <h2>Candidate stream.</h2>
              </div>
              <span>{history.length} evaluated</span>
            </div>
            <p className="nf-loop-muted">Read the change beside the score. Candidates train briefly and are ranked by a fitness score on the validation split (accuracy, precision and F1, minus size and latency); the sealed test split is only used after full training.</p>

            {/* Filter Pills */}
            <div className="nf-candidate-filters">
              <button className={candidateFilter === 'all' ? 'is-active' : ''} onClick={() => setCandidateFilter('all')}>
                All ({history.length})
              </button>
              <button className={candidateFilter === 'top' ? 'is-active' : ''} onClick={() => setCandidateFilter('top')}>
                Top 5
              </button>
              {lastGeneration > 1 && (
                <button className={candidateFilter === 'latest' ? 'is-active' : ''} onClick={() => setCandidateFilter('latest')}>
                  Gen {lastGeneration} only
                </button>
              )}
            </div>

            <div className="nf-candidate-list">
              {filteredCandidates.map((candidate, index) => {
                const parent = candidate.parent_id ? history.find((item) => item.candidate_id === candidate.parent_id) : null;
                const delta = parent && Number.isFinite(Number(candidate.accuracy)) && Number.isFinite(Number(parent.accuracy))
                  ? Number(candidate.accuracy) - Number(parent.accuracy)
                  : null;
                const accuracy = Number.isFinite(Number(candidate.accuracy)) ? `${Number(candidate.accuracy).toFixed(1)}%` : '—';
                const isChampion = candidate.candidate_id === (searching ? currentLeader?.candidate_id : championId);

                return (
                  <article className={`nf-candidate ${isChampion ? 'is-champion-blueprint' : ''}`} key={`${candidate.candidate_id || 'candidate'}-${index}`}>
                    <div className="nf-candidate-copy">
                      <div className="flex items-center">
                        <strong>{formatCandidateDisplayTitle(candidate.candidate_id, index, candidate.generation)}</strong>
                        {isChampion && <span className="nf-candidate-crown-tag">👑 {searching ? 'CURRENT LEADER' : 'SEARCH WINNER'}</span>}
                      </div>
                      <span>
                        gen {candidate.generation || '—'} · {candidate.parent_id ? `parent ${candidate.parent_id.split('-').at(-1)}` : 'seed'}
                        <small className="opacity-60 font-mono text-[9px] ml-2">({candidate.candidate_id})</small>
                      </span>
                      <p>{describeMutations(candidate)}</p>
                      {delta !== null && (
                        <small className={delta >= 0 ? 'nf-candidate-delta is-up' : 'nf-candidate-delta is-down'}>
                          vs parent {delta >= 0 ? '+' : ''}{delta.toFixed(1)} percentage points
                        </small>
                      )}
                    </div>
                    <div className="nf-candidate-score">
                      <span>validation acc</span>
                      <b>{accuracy}</b>
                      <small>F1 {Number.isFinite(Number(candidate.f1)) ? `${Number(candidate.f1).toFixed(1)}%` : '—'}</small>
                      <small>{formatParamCount(candidate.parameters)}</small>
                    </div>
                  </article>
                );
              })}
              {!history.length && <div className="nf-loop-empty">Candidates will appear here as the proxy loop evaluates them.</div>}
            </div>
          </div>

          <div className="nf-loop-panel nf-loop-train">
            <div className="nf-loop-panel-heading">
              <div>
                <p className="nf-kicker">03 / ship</p>
                <h2>Forge the winner.</h2>
              </div>
              <Zap size={20} className={training ? 'text-amber-400 animate-bounce' : 'text-zinc-400'} />
            </div>
            <p className="nf-loop-muted">
              Proxy search selects a blueprint. Full training turns that blueprint into the production checkpoint used by live inference.
            </p>

            {/* Active Blueprint Specification Box */}
            {champion?.config && (
              <div className="nf-blueprint-receipt">
                <div className="nf-blueprint-header">
                  <Layers size={13} />
                  <span>SEARCH WINNER BLUEPRINT · {champion.candidate_id?.split('-').slice(-2).join(' ')}</span>
                </div>
                <div className="nf-blueprint-grid">
                  <div><span>Layers</span><strong>{champion.config.num_layers} {champion.config.block_type && champion.config.block_type !== 'plain' ? `${champion.config.block_type} ` : ''}blocks</strong></div>
                  <div><span>Filters</span><strong>{JSON.stringify(champion.config.filters)}</strong></div>
                  <div><span>Activation</span><strong>{champion.config.activation}</strong></div>
                  <div><span>Params</span><strong>{formatParamCount(champion.parameters)}</strong></div>
                </div>
              </div>
            )}

            <label className="nf-training-label">
              Full-training epochs
              <input type="number" min="1" max="100" value={trainingEpochs} disabled={training} onChange={(event) => setTrainingEpochs(event.target.value)} />
            </label>

            {/* Active Deep Training Furnace */}
            {training ? (
              <div className="nf-deep-train-active">
                <div className="nf-deep-train-meta">
                  <RotateCcw size={13} className="nf-spin text-amber-400" />
                  <strong>DEEP FORGE IN PROGRESS</strong>
                  <span>up to {trainingEpochs} epochs · validation picks the checkpoint</span>
                </div>
                <div className="nf-deep-train-bar">
                  <div className="nf-deep-train-fill" style={liveBest ? { width: `${Math.round((liveCurve[liveCurve.length - 1].epoch / jobProgress[jobProgress.length - 1].total_epochs) * 100)}%`, animation: 'none' } : undefined} />
                </div>
                <small className="nf-deep-train-note">
                  {liveBest
                    ? `Epoch ${liveCurve[liveCurve.length - 1].epoch} of up to ${jobProgress[jobProgress.length - 1].total_epochs}: best validation F1 ${liveBest.valid_f1.toFixed(1)}% at epoch ${liveBest.epoch}. Stops early when validation stops improving.`
                    : (trainingMessage || 'Loading data and building the model…')}
                </small>
              </div>
            ) : (
              <button 
                className="nf-loop-button nf-loop-button-dark" 
                disabled={!dataset || baselineBusy || !hasSearchResults || !champion?.config}
                onClick={() => trainChampion(championId)}
              >
                {hasSearchResults ? 'Train the search winner' : 'Run a search on this dataset first'}
              </button>
            )}

            {trainedForActiveDataset && !training && (
              <div className="nf-trained-success-banner">
                <Check size={14} />
                <span>{activeIsChampion
                  ? 'Search winner trained and serving live inference'
                  : `Serving ${formatCandidateDisplayTitle(activeCandidateId, 0)} from an earlier run; the current search winner is not trained yet`}</span>
              </div>
            )}

            {trainedForActiveDataset && (
              <div className="nf-loop-final-metrics">
                <span>test accuracy <b>{Number.isFinite(latest.accuracy) ? `${latest.accuracy.toFixed(1)}%` : '—'}</b></span>
                <span>test macro F1 <b>{Number.isFinite(latest.f1) ? `${latest.f1.toFixed(1)}%` : '—'}</b></span>
              </div>
            )}

            {trainedForActiveDataset && (
              <button 
                className="nf-loop-button nf-loop-button-crucible"
                onClick={() => { window.location.href = '/#crucible'; }}
              >
                <Zap size={14} /> Test Specimen in Live Crucible →
              </button>
            )}

            {trainedForActiveDataset && (
              <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
                {status?.active_model?.bundle_id && (
                  <a className="nf-loop-download" href="/api/download/bundle" download>
                    <Download size={14} /> Download full bundle (.zip)
                  </a>
                )}
                <a className="nf-loop-download" href="/api/download/weights" download>
                  <Download size={14} /> Weights only
                </a>
              </div>
            )}
          </div>
        </section>

        <section id="nf-loop-evidence" className="nf-loop-evidence" aria-label="Search lineage and training evidence" style={{ display: 'grid', gap: 18, marginTop: 28, marginBottom: 28 }}>
          <LineageBay
            history={hasSearchResults ? history : []}
            winnerId={liveCandidates.length ? null : (historyPayload.search?.best_candidate_id)}
            manualBaseline={hasSearchResults ? historyPayload.manual_baseline : null}
            numClasses={dataset?.num_classes}
            busy={training || searching || resnetTraining}
            onTrain={(id, epochs) => {
              document.getElementById('nf-loop-phase-ship')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
              trainChampion(id, epochs);
            }}
          />
          <TrainingEvidenceBay
            active={trainedForActiveDataset ? status?.active_model : null}
            liveCurve={(training || resnetTraining) ? liveCurve : null}
            liveBestEpoch={liveBest?.epoch}
          />
        </section>

        <section className="nf-loop-panel nf-loop-comparison-panel nf-scroll-reveal" aria-labelledby="comparison-heading" data-scroll-reveal="loop-compare">
          <div className="nf-loop-panel-heading"><div><p className="nf-kicker">04 / compare</p><h2 id="comparison-heading">Put the models side by side.</h2></div><BarChart3 size={20} /></div>
          <p className="nf-loop-muted">Every model below is trained with the same split, seed and protocol, and scored once on the sealed test split. The two CNNs learn from scratch; the other four start from ImageNet-pretrained weights. Accuracy, precision, recall and F1 are percentages; MSE and RMSE compare probability vectors with one-hot targets, so lower is better.</p>
          {comparisonError && <div className="nf-loop-error">{comparisonError}</div>}
          <div className="nf-comparison-table-wrap">
            <table className="nf-comparison-table">
              <thead><tr><th>Model</th>{comparisonMetricKeys.map((key) => <th key={key}>{key.toUpperCase()}</th>)}<th>State</th></tr></thead>
              <tbody>
                {comparisonLoading && <tr><td colSpan="8">Reading saved comparison evidence…</td></tr>}
                {!comparisonLoading && comparisonModels.map((model) => <tr key={model.id}>
                  <th><strong>{model.label}</strong><small>{model.kind}<br />{model.budget}</small></th>
                  {comparisonMetricKeys.map((key) => <td key={key} className={bestByMetric[key] === model.id ? 'is-best' : undefined}>{formatMetric(model.metrics?.[key], key)}</td>)}
                  <td><span className={`nf-comparison-status nf-comparison-status-${model.status}`}>{model.status === 'not_run' ? 'not run' : model.status}</span></td>
                </tr>)}
                {!comparisonLoading && !comparisonModels.length && <tr><td colSpan="8">Mount a dataset to create a comparison context.</td></tr>}
              </tbody>
            </table>
          </div>
          <div className="nf-comparison-actions">
            <div>
              <p className="nf-kicker">Industry baselines / transfer learning</p>
              <h3>Train a baseline on the same split.</h3>
              <p>The manual CNN trains from scratch. The pretrained models keep their ImageNet backbone frozen and learn a new classifier head at 224px. Baselines never replace the champion checkpoint.</p>
              {resnetTraining && (
                <p className="nf-loop-muted" aria-live="polite">
                  {liveBest
                    ? `Epoch ${liveCurve[liveCurve.length - 1].epoch} of up to ${jobProgress[jobProgress.length - 1].total_epochs}: best validation F1 ${liveBest.valid_f1.toFixed(1)}%.`
                    : 'Loading data and pretrained weights…'}
                </p>
              )}
            </div>
            <div className="nf-comparison-run">
              <label>
                Baseline
                <select
                  className="nf-loop-select"
                  value={selectedTransferModel}
                  disabled={baselineBusy}
                  onChange={(event) => setSelectedTransferModel(event.target.value)}
                >
                  {transferOptions.map((opt) => (
                    <option key={opt.id} value={opt.id}>
                      {opt.fullName}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Epochs
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={resnetEpochs}
                  disabled={resnetTraining}
                  onChange={(event) => setResnetEpochs(event.target.value)}
                />
              </label>
              <button
                className="nf-loop-button nf-loop-button-accent"
                disabled={!dataset || baselineBusy}
                onClick={runTransferComparison}
              >
                {resnetTraining ? (
                  <><RotateCcw className="nf-spin" size={15} /> Comparing…</>
                ) : (
                  `Train ${transferOptions.find((o) => o.id === selectedTransferModel)?.label || 'Baseline'}`
                )}
              </button>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginTop: '12px' }}>
            {transferOptions.map((opt) => {
              const readyModel = comparisonModels.find((m) => m.id === opt.id && m.status === 'ready');
              if (!readyModel) return null;
              return (
                <a
                  key={opt.id}
                  className="nf-loop-download"
                  href={`/api/download/transfer-weights?model=${opt.id}`}
                  download
                >
                  <Download size={14} /> Download {readyModel.label} checkpoint
                </a>
              );
            })}
          </div>
          <div className="nf-comparison-footnote"><Info size={14} /> {comparison?.metric_definitions?.mse || 'MSE/RMSE are not regression targets here; they summarize probability-vector error.'}</div>
        </section>

        <section id="nf-loop-sequencer" style={{ marginBottom: 28 }}>
          <ArchitectureSequencer
            initialConfig={champion?.config || historyPayload.config}
            numClasses={dataset?.num_classes || 5}
            championAcc={historyPayload.search?.best_metrics?.accuracy || 0}
            proxyEpochs={historyPayload.search?.proxy_epochs || 3}
          />
        </section>

        <section className="nf-loop-panel nf-saved-runs" aria-labelledby="saved-runs-heading">
          <div className="nf-loop-panel-heading nf-saved-runs-heading">
            <div><p className="nf-kicker">local run shelf / this browser only</p><h2 id="saved-runs-heading">Keep a record of the winners.</h2></div>
            <button className="nf-loop-button" disabled={!dataset || !champion} onClick={saveCurrentRun}><Save size={15} /> Save current champion</button>
          </div>
          <p className="nf-loop-muted">Saved snapshots stay in this browser and include scores, parameter count, and dataset size, never the local folder path. Proxy and held-out scores are labelled separately; compare runs on the same dataset and split.</p>
          {savedRuns.length ? <div className="nf-saved-runs-table-wrap"><table className="nf-saved-runs-table">
            <thead><tr><th>Run</th><th>Accuracy</th><th>Macro F1</th><th>Parameters</th><th>Latency</th><th>Evidence</th><th><span className="sr-only">Remove</span></th></tr></thead>
            <tbody>{savedRuns.map((run) => <tr key={run.id}>
              <th><strong>{run.candidateId}</strong><small>{run.datasetLabel} · gen {run.generation || '—'} · {new Date(run.createdAt).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}</small></th>
              <td>{formatMetric(run.accuracy, 'accuracy')}</td>
              <td>{formatMetric(run.f1, 'f1')}</td>
              <td>{Number.isFinite(Number(run.parameters)) ? Number(run.parameters).toLocaleString() : '—'}</td>
              <td>{Number.isFinite(Number(run.latencyMs)) ? `${Number(run.latencyMs).toFixed(1)} ms` : '—'}</td>
              <td><span className={`nf-comparison-status ${run.scoreBasis === 'held-out test' ? 'nf-comparison-status-ready' : 'nf-comparison-status-not_run'}`}>{run.scoreBasis}</span></td>
              <td><button className="nf-saved-run-remove" onClick={() => removeSavedRun(run.id)} aria-label={`Remove saved run ${run.candidateId}`}><Trash2 size={14} /></button></td>
            </tr>)}</tbody>
          </table></div> : <div className="nf-saved-runs-empty">No saved runs yet. Search for a candidate, then save the current champion here.</div>}
        </section>
      </main>
      <footer className="nf-footer"><span>NEUROFORGE / mutation loop</span><span>build · mutate · ship</span></footer>
    </div>
  );
}
