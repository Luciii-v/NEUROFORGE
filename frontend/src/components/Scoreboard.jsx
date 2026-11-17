import { useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { api, compact, pct } from '../api';

function MetricCell({ value }) {
  if (value === null || value === undefined) return <span className="muted">—</span>;
  return (
    <div className="metric-cell">
      <span className="num">{pct(value)}</span>
      <div className="track" aria-hidden="true"><div style={{ width: `${Math.max(0, Math.min(100, value))}%` }} /></div>
    </div>
  );
}

export default function Scoreboard({ comparison, busy, onStart, onChange }) {
  const [epochs, setEpochs] = useState(25);
  const [working, setWorking] = useState('');
  const [error, setError] = useState('');
  const rows = comparison?.models || [];
  const ready = rows.filter((r) => r.status === 'ready');
  const best = ready.length ? ready.reduce((a, b) => ((b.metrics?.f1 ?? -1) > (a.metrics?.f1 ?? -1) ? b : a)) : null;
  const winner = rows.find((r) => r.id === 'dynamic_cnn');
  const sampleCount = ready[0]?.sample_count;

  const train = async (row) => {
    setWorking(row.id);
    setError('');
    try {
      const request = api('/api/train/baseline', { json: { model_type: row.id, epochs: Number(epochs) } });
      onStart?.();
      await request;
      await onChange();
    } catch (e) {
      setError(e.message);
    } finally {
      setWorking('');
    }
  };

  return (
    <section id="step-compare" className="section" aria-labelledby="compare-title">
      <div className="section-head">
        <h2 id="compare-title">Compare</h2>
        <p>Every row uses the same split, seed and training protocol, so the numbers are directly comparable. Models that haven't been trained on this dataset show no score.</p>
        <div className="actions">
          <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>Baseline epochs
            <input type="number" min="1" max="200" value={epochs} onChange={(e) => setEpochs(e.target.value)} />
          </label>
        </div>
      </div>
      {error && <div className="notice error" role="alert" style={{ marginBottom: 12 }}>{error}</div>}
      <div className="panel score-scroll">
        <table className="score-table">
          <caption className="sr-only">Model comparison on the test split</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Test macro F1</th>
              <th scope="col">Test accuracy</th>
              <th scope="col" className="num">Valid F1</th>
              <th scope="col" className="num">Parameters</th>
              <th scope="col" className="num">Latency</th>
              <th scope="col"><span className="sr-only">Action</span></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} className={row.id === 'dynamic_cnn' ? 'winner' : undefined}>
                <th scope="row" style={{ textAlign: 'left', fontWeight: 600 }}>
                  {row.label}
                  <div className="muted small" style={{ fontWeight: 400 }}>{row.kind}{row.budget ? `, ${row.budget}` : ''}</div>
                  {row.status === 'ready' && !row.test_is_independent && (
                    <div className="small" style={{ color: 'var(--sodium)', fontWeight: 500, display: 'flex', gap: 6, alignItems: 'center' }}>
                      <AlertTriangle size={13} aria-hidden="true" />Legacy run: test also used for selection
                    </div>
                  )}
                </th>
                <td><MetricCell value={row.metrics?.f1} /></td>
                <td><MetricCell value={row.metrics?.accuracy} /></td>
                <td className="num">{pct(row.validation?.f1)}</td>
                <td className="num">
                  {compact(row.parameters)}
                  {row.trainable_parameters && row.trainable_parameters !== row.parameters && (
                    <div className="muted small">{compact(row.trainable_parameters)} trained</div>
                  )}
                </td>
                <td className="num">{row.latency_ms ? `${row.latency_ms.toFixed(1)} ms` : '—'}</td>
                <td className="num">
                  {row.id !== 'dynamic_cnn' && (
                    <button type="button" className="btn small" disabled={busy || !!working || !comparison?.dataset} onClick={() => train(row)}>
                      {working === row.id ? 'Training…' : row.status === 'ready' ? 'Retrain' : 'Train'}
                    </button>
                  )}
                  {row.status === 'other_dataset' && <div className="muted small">trained on another dataset</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {best && winner?.metrics && best.id !== 'dynamic_cnn' && (
        <div className="notice" style={{ marginTop: 12 }}>
          {best.label} leads with {pct(best.metrics.f1)} test F1, against {pct(winner.metrics.f1)} for the evolved CNN.
          {best.kind.includes('pretrained')
            ? ' A pretrained backbone already knows edges, textures and shapes from 1.2M ImageNet photos; a CNN evolved from scratch has to learn them from this dataset alone, which needs far more images.'
            : ''}
        </div>
      )}
      {sampleCount && <p className="muted small">Test split: {sampleCount} samples. Latency is per sample on this machine, batched.</p>}
    </section>
  );
}
