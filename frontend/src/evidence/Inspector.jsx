import { Crown, Hammer } from 'lucide-react';
import { compact, describeConfig, describeOperation, pct } from './api';
import ArchitectureDrawing from './ArchitectureDrawing';

export default function Inspector({ candidate, winnerId, baseline, numClasses, epochs, setEpochs, onTrain, busy, onSelect }) {
  if (!candidate) {
    return (
      <aside className="panel panel-pad inspector">
        <h3>Candidate inspector</h3>
        <p className="muted small" style={{ margin: 0 }}>Select any node in the lineage to see its blueprint, what changed from its parent and how it scored.</p>
      </aside>
    );
  }
  const isWinner = candidate.candidate_id === winnerId;
  const ops = candidate.mutation_operations || [];
  const delta = baseline ? candidate.f1 - baseline.f1 : null;

  return (
    <aside className="panel panel-pad inspector" aria-live="polite">
      <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0 }}>Generation {candidate.generation}, candidate {Number(candidate.candidate_id.split('-C').pop())}</h3>
        {isWinner && <span className="tag sodium"><Crown size={13} aria-hidden="true" />Search winner</span>}
      </div>
      <p className="muted small" style={{ margin: '4px 0 10px' }}>{describeConfig(candidate.config)}</p>

      <ArchitectureDrawing config={candidate.config} numClasses={numClasses} />

      <dl className="kv">
        <dt>Validation F1</dt><dd><strong>{pct(candidate.f1)}</strong>{delta !== null && <span className="muted"> ({delta >= 0 ? '+' : ''}{delta.toFixed(1)} vs manual CNN)</span>}</dd>
        <dt>Validation accuracy</dt><dd>{pct(candidate.accuracy)}</dd>
        <dt>Search score</dt><dd>{candidate.score.toFixed(1)}</dd>
        <dt>Parameters</dt><dd>{compact(candidate.parameters)}</dd>
        <dt>Kernels</dt><dd>{candidate.config.kernel_sizes.map((k) => `${k}×${k}`).join(', ')}</dd>
        <dt>Regularisation</dt><dd>dropout {candidate.config.dropout}, BatchNorm {candidate.config.batch_norm ? 'on' : 'off'}, pooling {candidate.config.pooling ? 'on' : 'off'}</dd>
        {candidate.evaluation_reused && (<><dt>Evaluation</dt><dd>reused (identical design seen earlier)</dd></>)}
      </dl>

      <h3 style={{ marginTop: 14 }}>{candidate.parent_id ? 'Mutations from parent' : 'Mutations from the seed design'}</h3>
      {ops.length ? (
        <ul className="mutations">
          {ops.map((op, i) => {
            const [label, change] = describeOperation(op);
            return <li key={i}><span>{label}</span><span>{change}</span></li>;
          })}
        </ul>
      ) : <p className="muted small" style={{ margin: 0 }}>None. This is the seed design.</p>}
      {candidate.parent_id && (
        <button type="button" className="btn small" style={{ marginTop: 10 }} onClick={() => onSelect(candidate.parent_id)}>Show parent</button>
      )}

      {onTrain && (<>
      <div className="row" style={{ marginTop: 18 }}>
        <label className="field">Training epochs
          <input type="number" min="1" max="200" value={epochs} onChange={(e) => setEpochs(e.target.value)} />
        </label>
        <button type="button" className="btn primary" disabled={busy} onClick={() => onTrain(candidate.candidate_id)}>
          <Hammer size={16} aria-hidden="true" />Train this design
        </button>
      </div>
      <p className="muted small" style={{ margin: '8px 0 0' }}>Trains from fresh weights. The checkpoint is chosen on validation, then the sealed test split is measured once.</p>
      </>)}
    </aside>
  );
}
