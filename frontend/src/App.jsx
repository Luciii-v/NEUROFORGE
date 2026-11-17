import { useCallback, useEffect, useState } from 'react';
import { Moon, Sun } from 'lucide-react';
import { api, datasetName, pct } from './api';
import DataSection from './components/DataSection';
import EvolveSection from './components/EvolveSection';
import Scoreboard from './components/Scoreboard';
import ShipSection from './components/ShipSection';
import TestBench from './components/TestBench';
import TrainSection from './components/TrainSection';

const sameRoot = (a, b) => !!a && !!b && a.replace(/\/+$/, '') === b.replace(/\/+$/, '');

function readTheme() {
  const fromUrl = new URLSearchParams(window.location.search).get('theme');
  if (fromUrl === 'light' || fromUrl === 'dark') return fromUrl;
  try {
    return localStorage.getItem('nf-theme') || 'dark';
  } catch {
    return 'dark';
  }
}

function Mark() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="3" fill="var(--sodium)" />
      <path d="M9 24V8l14 16V8" fill="none" stroke="var(--sodium-ink)" strokeWidth="3" strokeLinejoin="round" />
    </svg>
  );
}

export default function App() {
  const [theme, setTheme] = useState(readTheme);
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState({ history: [] });
  const [comparison, setComparison] = useState(null);
  const [offline, setOffline] = useState(false);
  const [trainError, setTrainError] = useState('');
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('nf-theme', theme); } catch { /* private mode */ }
  }, [theme]);

  const refresh = useCallback(async () => {
    try {
      const [s, h, c] = await Promise.all([api('/api/status'), api('/api/history'), api('/api/comparison')]);
      setStatus(s);
      setHistory(h);
      setComparison(c);
      setOffline(false);
      setRefreshKey((k) => k + 1);
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll only while the server is busy, so training progress streams in.
  const running = status?.job?.running;
  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(async () => {
      try {
        const s = await api('/api/status');
        setStatus(s);
        if (!s.job.running) refresh();
      } catch { setOffline(true); }
    }, 1500);
    return () => clearInterval(id);
  }, [running, refresh]);

  // A long request just started on the server: fetch status once so polling kicks in.
  const kick = () => setTimeout(async () => {
    try { setStatus(await api('/api/status')); } catch { /* next refresh will report */ }
  }, 400);

  const train = async (candidateId, epochs) => {
    setTrainError('');
    document.getElementById('step-train')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    const request = api('/api/train/full', { json: { candidate_id: candidateId, epochs } });
    kick();
    try {
      await request;
    } catch (e) {
      setTrainError(`Training failed: ${e.message}`);
    }
    refresh();
  };

  const workspace = status?.workspace || {};
  const dataset = workspace.dataset_metadata;
  const active = status?.active_model;
  const latestSearch = status?.latest_search;
  const searchMatches = sameRoot(latestSearch?.dataset?.dataset_root, dataset?.dataset_root);
  const busy = !!status?.job?.running;

  const steps = [
    { id: 'step-data', t: 'Data', s: dataset ? `${dataset.num_classes} classes` : 'mount a dataset', done: !!dataset },
    { id: 'step-evolve', t: 'Evolve', s: searchMatches ? `${history.history.length} designs` : 'run a search', done: searchMatches },
    { id: 'step-train', t: 'Train', s: active?.ready && workspace.model_matches_dataset ? `${pct(active.test_metrics?.f1)} test F1` : 'train the winner', done: !!(active?.ready && workspace.model_matches_dataset) },
    { id: 'step-test', t: 'Test', s: 'try samples', done: false },
    { id: 'step-ship', t: 'Ship', s: active?.bundle_id ? 'bundle ready' : 'export', done: !!active?.bundle_id },
  ];
  const current = steps.findIndex((step) => !step.done);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="wordmark"><Mark />NEUROFORGE</div>
        <div className="meta">
          {busy && <span className="job-pill"><span className="pulse" aria-hidden="true" />{status.job.label}</span>}
          <span>Dataset <strong>{dataset ? datasetName(dataset) : 'none'}</strong></span>
          <span>Device <strong>{status?.device || '…'}</strong></span>
          <button type="button" className="btn small" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
            {theme === 'dark' ? <Sun size={14} aria-hidden="true" /> : <Moon size={14} aria-hidden="true" />}
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>
      </header>

      <nav className="rail" aria-label="Workflow">
        {steps.map((step, i) => (
          <button key={step.id} type="button" className={step.done ? 'done' : i === current ? 'current' : ''}
            onClick={() => document.getElementById(step.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
            aria-current={i === current ? 'step' : undefined}>
            <span className="n">{i + 1}</span>
            <span><span className="t">{step.t}</span><span className="s">{step.s}</span></span>
          </button>
        ))}
      </nav>

      {offline && (
        <div className="notice error" role="alert" style={{ marginTop: 16 }}>
          Can't reach the studio server. Start it with <code>./run_studio.sh</code> and reload.
        </div>
      )}

      <main>
        <DataSection dataset={dataset} mounted={workspace.dataset_mounted} busy={busy} onChange={refresh} />
        <EvolveSection dataset={dataset} savedHistory={history.history || []} latestSearch={latestSearch}
          manualBaseline={searchMatches ? history.manual_baseline : null} searchMatchesDataset={searchMatches}
          busy={busy} theme={theme} onSearchFinished={refresh} onTrain={train} />
        <TrainSection active={active} job={status?.job} matchesDataset={workspace.model_matches_dataset} trainError={trainError} />
        <Scoreboard comparison={comparison} busy={busy} onStart={kick} onChange={refresh} />
        <TestBench ready={active?.ready} modelKey={`${active?.bundle_id}|${dataset?.dataset_root}`} />
        <ShipSection active={active} refreshKey={refreshKey} />
      </main>
    </div>
  );
}
