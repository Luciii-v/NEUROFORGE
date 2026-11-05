import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, BarChart3, Database, Download, Info, Moon, Play, RotateCcw, Sun, Trash2, UploadCloud, Zap } from 'lucide-react';

const initialSearch = { generations: 3, population: 4, proxyEpochs: 2 };

export default function MutationLoopPage({ theme, onToggleTheme, onReturn, scrollReady, scrollProgress, navScrolled }) {
  const [status, setStatus] = useState(null);
  const [historyPayload, setHistoryPayload] = useState({ history: [], metadata: {} });
  const [path, setPath] = useState('');
  const [busyDataset, setBusyDataset] = useState(false);
  const [datasetError, setDatasetError] = useState('');
  const [search, setSearch] = useState(initialSearch);
  const [searching, setSearching] = useState(false);
  const [searchStage, setSearchStage] = useState('ready');
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

  const workspace = status?.workspace || {};
  const dataset = workspace.dataset_metadata || historyPayload.metadata?.dataset || null;
  const history = liveCandidates.length ? liveCandidates : (historyPayload.history || []);
  const generations = useMemo(() => {
    const byGeneration = new Map();
    history.forEach((candidate) => {
      const generation = Number(candidate.generation);
      if (!generation) return;
      const current = byGeneration.get(generation);
      if (!current || Number(candidate.accuracy || 0) > Number(current.accuracy || 0)) {
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
      }
      if (message.type === 'generation_completed') {
        setCurrentGeneration(Number(message.generation));
        setSearchStage(`generation-${message.generation}`);
      }
      if (message.type === 'search_completed') {
        setSearchStage('complete');
        setSearching(false);
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

  const trainChampion = async () => {
    if (training || !dataset) return;
    setTraining(true);
    setTrainingMessage(`Training the selected architecture for ${trainingEpochs} epochs…`);
    try {
      const result = await readResponse(await fetch('/api/train/full', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs: Number(trainingEpochs) }),
      }));
      const testAccuracy = result.metrics?.held_out_test_metrics?.accuracy;
      setTrainingMessage(`Checkpoint saved. Test accuracy: ${testAccuracy?.toFixed?.(1) || '—'}%.`);
      await refresh();
    } catch (error) {
      setTrainingMessage(`Training failed: ${error.message}`);
    } finally {
      setTraining(false);
    }
  };

  const runResnetComparison = async () => {
    if (resnetTraining || !dataset) return;
    setResnetTraining(true);
    setComparisonError('');
    try {
      const result = await readResponse(await fetch('/api/train/resnet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs: Number(resnetEpochs) }),
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

        <section id="nf-loop-phase-build" className="nf-loop-grid nf-loop-intake nf-scroll-reveal" data-scroll-reveal="loop-build">
          <div className="nf-loop-panel nf-loop-panel-wide">
            <div className="nf-loop-panel-heading"><div><p className="nf-kicker">01 / build</p><h2>Mount the dataset.</h2></div><Database size={20} /></div>
            <p className="nf-loop-muted">Use a local ImageFolder path or upload a ZIP. Removing a local path only unmounts it; uploaded extractions can be deleted from NEUROFORGE’s workspace.</p>
            <div className="nf-loop-input-row"><input value={path} onChange={(event) => setPath(event.target.value)} placeholder="/path/to/image-folder" /><button className="nf-loop-button" onClick={mountPath} disabled={busyDataset}>{busyDataset ? 'Reading…' : 'Mount path'}</button></div>
            <div className="nf-loop-upload" onClick={() => uploadRef.current?.click()}><input ref={uploadRef} type="file" accept=".zip" onChange={(event) => uploadDataset(event.target.files?.[0])} /><UploadCloud size={20} /><strong>Upload dataset ZIP</strong><span>ImageFolder or Roboflow single-label export</span></div>
            {datasetError && <div className="nf-loop-error">{datasetError}</div>}
          </div>
          <div className="nf-loop-panel nf-loop-spec-card">
            <div className="nf-loop-panel-heading"><div><p className="nf-kicker">active specimen</p><h2>{dataset ? 'Dataset mounted' : 'Awaiting data'}</h2></div><span className="nf-loop-live-dot" /></div>
            {dataset ? <><strong className="nf-loop-dataset-name">{dataset.modality === 'audio' ? 'Audio classification' : 'Image classification'}</strong><div className="nf-loop-stat-list"><span>samples <b>{dataset.num_samples}</b></span><span>classes <b>{dataset.num_classes}</b></span><span>format <b>{dataset.format}</b></span></div><div className="nf-loop-tags">{dataset.class_names?.map((name) => <span key={name}>{name}</span>)}</div><button className="nf-loop-remove" onClick={removeDataset} disabled={busyDataset}><Trash2 size={14} /> {workspace.dataset_owned ? 'Delete uploaded extraction' : 'Unmount local path'}</button></> : <div className="nf-loop-empty">Nothing can mutate until a labelled dataset is mounted.</div>}
          </div>
        </section>

        <section id="nf-loop-phase-mutate" className="nf-loop-panel nf-loop-search-panel nf-scroll-reveal" data-scroll-reveal="loop-mutate">
          <div className="nf-loop-panel-heading"><div><p className="nf-kicker">02 / mutate</p><h2>Set the search budget.</h2></div><span className="nf-loop-search-state">{searching ? <><RotateCcw className="nf-spin" size={14} /> {searchStage}</> : searchStage}</span></div>
          <div className="nf-loop-controls">
            <label>Generations<input type="number" min="1" max="10" value={search.generations} disabled={searching} onChange={(event) => setSearch({ ...search, generations: event.target.value })} /></label>
            <label>Population / generation<input type="number" min="2" max="16" value={search.population} disabled={searching} onChange={(event) => setSearch({ ...search, population: event.target.value })} /></label>
            <label>Proxy epochs / candidate<input type="number" min="1" max="5" value={search.proxyEpochs} disabled={searching} onChange={(event) => setSearch({ ...search, proxyEpochs: event.target.value })} /></label>
            <button className="nf-loop-button nf-loop-button-accent" onClick={startSearch} disabled={searching || !dataset}><Play size={15} fill="currentColor" /> {searching ? 'Evolving…' : 'Start mutation search'}</button>
          </div>
          <div className="nf-generation-strip">{Array.from({ length: Math.max(3, Number(search.generations) || 3) }, (_, index) => { const gen = index + 1; const record = generations.find(([number]) => number === gen)?.[1]; return <div className={currentGeneration === gen ? 'is-current' : ''} key={gen}><span>GEN {String(gen).padStart(2, '0')}</span><strong>{record ? `${Number(record.accuracy).toFixed(1)}%` : '—'}</strong><small>{record?.mutation_operations?.[0]?.field || (gen === 1 ? 'seed population' : 'waiting for candidates')}</small></div>; })}</div>
        </section>

        <section id="nf-loop-phase-ship" className="nf-loop-lower-grid nf-scroll-reveal" data-scroll-reveal="loop-ship">
          <div className="nf-loop-panel nf-loop-feed"><div className="nf-loop-panel-heading"><div><p className="nf-kicker">live lineage</p><h2>Candidate stream.</h2></div><span>{history.length} evaluated</span></div><div className="nf-candidate-list">{history.slice(0, 12).map((candidate, index) => <div className="nf-candidate" key={`${candidate.candidate_id || 'candidate'}-${index}`}><div><strong>{candidate.candidate_id || `candidate-${index + 1}`}</strong><span>gen {candidate.generation || '—'} · {candidate.parent_id ? `parent ${candidate.parent_id.split('-').at(-1)}` : 'seed'}</span><small>{candidate.mutation_operations?.map((operation) => operation.field).join(' · ') || 'architecture evaluated'}</small></div><b>{Number(candidate.accuracy || 0).toFixed(1)}%</b></div>)}{!history.length && <div className="nf-loop-empty">Candidates will appear here as the proxy loop evaluates them.</div>}</div></div>
          <div className="nf-loop-panel nf-loop-train"><div className="nf-loop-panel-heading"><div><p className="nf-kicker">03 / ship</p><h2>Forge the winner.</h2></div><Zap size={20} /></div><p className="nf-loop-muted">Proxy search selects a blueprint. Full training turns that blueprint into the checkpoint used by live inference.</p><label className="nf-training-label">Full-training epochs<input type="number" min="1" max="100" value={trainingEpochs} disabled={training} onChange={(event) => setTrainingEpochs(event.target.value)} /></label><button className="nf-loop-button nf-loop-button-dark" disabled={!dataset || training || !historyPayload.config} onClick={trainChampion}>{training ? <><RotateCcw className="nf-spin" size={15} /> Training…</> : 'Train selected architecture'}</button>{trainingMessage && <div className="nf-training-message">{trainingMessage}</div>}<div className="nf-loop-final-metrics"><span>accuracy <b>{latest.accuracy ? `${latest.accuracy.toFixed(1)}%` : '—'}</b></span><span>macro F1 <b>{latest.f1 ? `${latest.f1.toFixed(1)}%` : '—'}</b></span></div><a className="nf-loop-download" href="/api/download/weights" download><Download size={14} /> Download checkpoint</a></div>
        </section>

        <section className="nf-loop-panel nf-loop-comparison-panel nf-scroll-reveal" aria-labelledby="comparison-heading" data-scroll-reveal="loop-compare">
          <div className="nf-loop-panel-heading"><div><p className="nf-kicker">04 / compare</p><h2 id="comparison-heading">Put the models side by side.</h2></div><BarChart3 size={20} /></div>
          <p className="nf-loop-muted">The mutated DynamicCNN, the fixed manual CNN, and ResNet-18 are separate evidence tracks. Accuracy, precision, recall and F1 are percentages; MSE and RMSE compare probability vectors with one-hot targets, so lower is better.</p>
          {comparisonError && <div className="nf-loop-error">{comparisonError}</div>}
          <div className="nf-comparison-table-wrap">
            <table className="nf-comparison-table">
              <thead><tr><th>Model</th>{comparisonMetricKeys.map((key) => <th key={key}>{key.toUpperCase()}</th>)}<th>State</th></tr></thead>
              <tbody>
                {comparisonLoading && <tr><td colSpan="8">Reading saved comparison evidence…</td></tr>}
                {!comparisonLoading && comparisonModels.map((model) => <tr key={model.id}>
                  <th><strong>{model.label}</strong><small>{model.kind}<br />{model.budget}</small></th>
                  {comparisonMetricKeys.map((key) => <td key={key}>{formatMetric(model.metrics?.[key], key)}</td>)}
                  <td><span className={`nf-comparison-status nf-comparison-status-${model.status}`}>{model.status === 'not_run' ? 'not run' : model.status}</span></td>
                </tr>)}
                {!comparisonLoading && !comparisonModels.length && <tr><td colSpan="8">Mount a dataset to create a comparison context.</td></tr>}
              </tbody>
            </table>
          </div>
          <div className="nf-comparison-actions">
            <div><p className="nf-kicker">ResNet-18 / fixed reference</p><h3>Run the transfer-learning comparison.</h3><p>Uses the same active image dataset and declared full-training budget. The backbone is frozen; only the replacement classifier head learns. This does not overwrite the selected DynamicCNN checkpoint.</p></div>
            <div className="nf-comparison-run"><label>ResNet epochs<input type="number" min="1" max="100" value={resnetEpochs} disabled={resnetTraining} onChange={(event) => setResnetEpochs(event.target.value)} /></label><button className="nf-loop-button nf-loop-button-accent" disabled={!dataset || resnetTraining} onClick={runResnetComparison}>{resnetTraining ? <><RotateCcw className="nf-spin" size={15} /> Comparing…</> : 'Train ResNet-18 baseline'}</button></div>
          </div>
          {comparisonModels.find((model) => model.id === 'resnet18')?.status === 'ready' && <a className="nf-loop-download" href="/api/download/resnet-weights" download><Download size={14} /> Download ResNet-18 comparison checkpoint</a>}
          <div className="nf-comparison-footnote"><Info size={14} /> {comparison?.metric_definitions?.mse || 'MSE/RMSE are not regression targets here; they summarize probability-vector error.'}</div>
        </section>
      </main>
      <footer className="nf-footer"><span>NEUROFORGE / mutation loop</span><span>build · mutate · ship</span></footer>
    </div>
  );
}
