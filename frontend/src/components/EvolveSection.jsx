import { useEffect, useMemo, useRef, useState } from 'react';
import { Play, Square } from 'lucide-react';
import { pct, searchSocket } from '../api';
import Inspector from './Inspector';
import LineageTree from './LineageTree';

export default function EvolveSection({ dataset, savedHistory, latestSearch, manualBaseline, searchMatchesDataset, busy, theme, onSearchFinished, onTrain }) {
  const [budget, setBudget] = useState({ generations: 5, population: 8, proxy_epochs: 3 });
  const [live, setLive] = useState(null); // candidates streamed during a run
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const [epochs, setEpochs] = useState(40);
  const socketRef = useRef(null);

  const history = live || (searchMatchesDataset ? savedHistory : []);
  const winnerId = useMemo(() => {
    if (!history.length) return null;
    if (!live && latestSearch?.best_candidate_id) return latestSearch.best_candidate_id;
    return history.reduce((best, c) => (c.score > best.score ? c : best), history[0]).candidate_id;
  }, [history, live, latestSearch]);
  const selected = history.find((c) => c.candidate_id === (selectedId || winnerId));
  const total = budget.generations * budget.population;
  const done = live?.length || 0;

  useEffect(() => () => socketRef.current?.close(), []);

  const start = () => {
    setError('');
    setLive([]);
    setSelectedId(null);
    setRunning(true);
    setMessage('Evaluating the manual baseline…');
    const socket = searchSocket();
    socketRef.current = socket;
    socket.onopen = () => socket.send(JSON.stringify(budget));
    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'candidate_evaluated') {
        setLive((prev) => [...(prev || []), msg.data]);
        setMessage(`Generation ${msg.data.generation}: evaluated ${msg.data.candidate_id.split('-').slice(-1)[0]}`);
      } else if (msg.type === 'search_completed') {
        setMessage(msg.stopped_early ? 'Search stopped. The best design so far is kept.' : 'Search complete.');
        setRunning(false);
        onSearchFinished().then(() => setLive(null));
      } else if (msg.type === 'error') {
        setError(msg.message);
        setRunning(false);
        setLive(null);
      }
    };
    socket.onerror = () => setError('Lost the connection to the studio server. Is it still running?');
    socket.onclose = () => setRunning(false);
  };

  const stop = () => {
    socketRef.current?.send(JSON.stringify({ type: 'stop' }));
    setMessage('Stopping after the current generation…');
  };

  const field = (key, label, min, max) => (
    <label className="field">{label}
      <input type="number" min={min} max={max} value={budget[key]} disabled={running}
        onChange={(e) => setBudget({ ...budget, [key]: Math.max(min, Math.min(max, Number(e.target.value) || min)) })} />
    </label>
  );

  return (
    <section id="step-evolve" className="section" aria-labelledby="evolve-title">
      <div className="section-head">
        <h2 id="evolve-title">Evolve</h2>
        <p>Each generation keeps the best design, then breeds mutated children from the top half. Every candidate trains briefly from scratch and is ranked on the validation split.</p>
      </div>

      <div className="panel panel-pad" style={{ marginBottom: 16 }}>
        <div className="budget">
          {field('generations', 'Generations', 1, 20)}
          {field('population', 'Designs per generation', 2, 24)}
          {field('proxy_epochs', 'Epochs per design', 1, 5)}
          <span className="total">{total} designs, {total * budget.proxy_epochs} short training runs</span>
          <span style={{ marginLeft: 'auto', display: 'flex', gap: 10 }}>
            {running ? (
              <button type="button" className="btn danger" onClick={stop}><Square size={15} aria-hidden="true" />Stop search</button>
            ) : (
              <button type="button" className="btn primary" onClick={start} disabled={!dataset || busy}>
                <Play size={15} aria-hidden="true" />Start search
              </button>
            )}
          </span>
        </div>
        {running && (
          <div style={{ marginTop: 14 }}>
            <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={done}>
              <div style={{ width: `${(done / total) * 100}%` }} />
            </div>
            <p className="small muted" style={{ margin: '6px 0 0' }} aria-live="polite">{message} ({done} of {total})</p>
          </div>
        )}
        {!running && message && <p className="small muted" style={{ margin: '10px 0 0' }}>{message}</p>}
        {!dataset && <p className="small muted" style={{ margin: '10px 0 0' }}>Mount a dataset first.</p>}
        {error && <div className="notice error" style={{ marginTop: 12 }} role="alert">{error}</div>}
      </div>

      <div className="evolve-grid">
        <div>
          <div className="lineage-wrap">
            {history.length ? (
              <LineageTree history={history} winnerId={winnerId} selectedId={selected?.candidate_id} onSelect={setSelectedId} theme={theme} />
            ) : (
              <div className="empty-lineage"><div>
                <strong>No lineage yet for this dataset.</strong>
                <p className="muted" style={{ marginBottom: 0 }}>Start a search and the family tree draws itself here as each design finishes, with the winning ancestry traced in yellow.</p>
              </div></div>
            )}
          </div>
          {history.length > 0 && (
            <div className="lineage-legend">
              <span><i style={{ width: 22, height: 2, background: 'var(--sodium)' }} />Winner's ancestry</span>
              <span><i style={{ width: 22, borderTop: '1.5px dashed var(--line-strong)' }} />Elite copied forward</span>
              <span><i style={{ width: 22, height: 10, background: 'linear-gradient(90deg, var(--heat-lo), var(--heat-hi))' }} />Validation F1, low to high (number on each node)</span>
              <span>Bars inside a node are its layers, taller means more filters</span>
              {manualBaseline && <span>Manual CNN baseline: {pct(manualBaseline.f1)} F1</span>}
            </div>
          )}
        </div>
        <Inspector candidate={selected} winnerId={winnerId} baseline={manualBaseline} numClasses={dataset?.num_classes}
          epochs={epochs} setEpochs={setEpochs} busy={busy || running}
          onSelect={setSelectedId} onTrain={(id) => onTrain(id, Number(epochs))} />
      </div>
    </section>
  );
}
