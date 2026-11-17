import { useMemo, useState } from 'react';
import { pct } from './api';
import { ConfusionMatrix, LearningCurve } from './Charts';
import Inspector from './Inspector';
import LineageTree from './LineageTree';
import './evidence.css';

/** Full family tree of a search with a candidate inspector beside it. */
export function LineageBay({ history = [], winnerId, manualBaseline, numClasses, onTrain, busy }) {
  const [selectedId, setSelectedId] = useState(null);
  const [epochs, setEpochs] = useState(40);
  const winner = useMemo(() => {
    if (!history.length) return null;
    return winnerId && history.some((c) => c.candidate_id === winnerId)
      ? winnerId
      : history.reduce((best, c) => (c.score > best.score ? c : best), history[0]).candidate_id;
  }, [history, winnerId]);
  const selected = history.find((c) => c.candidate_id === (selectedId || winner));

  return (
    <div className="nf-evidence nf-evidence-grid">
      <div className="nf-evidence-bay">
        <p className="nf-bay-label">lineage / every evaluated design, ranked on validation</p>
        <div className="lineage-wrap">
          {history.length ? (
            <LineageTree history={history} winnerId={winner} selectedId={selected?.candidate_id} onSelect={setSelectedId} />
          ) : (
            <div className="empty-lineage"><div>
              <strong>No lineage for this dataset yet.</strong>
              <p className="muted" style={{ marginBottom: 0 }}>Run a search and the family tree draws itself here, with the winner's ancestry traced in lime.</p>
            </div></div>
          )}
        </div>
        {history.length > 0 && (
          <div className="lineage-legend">
            <span><i style={{ width: 22, height: 2, background: 'var(--sodium)' }} />winner's ancestry</span>
            <span><i style={{ width: 22, borderTop: '1.5px dashed var(--line-strong)' }} />elite copied forward</span>
            <span><i style={{ width: 22, height: 10, background: 'linear-gradient(90deg, var(--heat-lo), var(--heat-hi))' }} />validation F1, low → high</span>
            <span>bars in a node = its layers (taller = more filters)</span>
            {manualBaseline && <span>manual CNN baseline {pct(manualBaseline.f1)} F1</span>}
          </div>
        )}
      </div>
      <Inspector candidate={selected} winnerId={winner} baseline={manualBaseline} numClasses={numClasses}
        epochs={epochs} setEpochs={setEpochs} busy={busy} onSelect={setSelectedId}
        onTrain={onTrain ? (id) => onTrain(id, Number(epochs)) : undefined} />
    </div>
  );
}

/** Learning curve of the active checkpoint plus its test-split confusion heatmap. */
export function TrainingEvidenceBay({ active, liveCurve, liveBestEpoch }) {
  const curve = liveCurve || active?.learning_curve;
  const bestEpoch = liveCurve ? liveBestEpoch : active?.best_epoch;
  const test = active?.test_metrics || {};
  const valid = active?.validation_metrics || {};
  if (!curve?.length && !test.confusion_matrix) return null;

  return (
    <div className="nf-evidence nf-evidence-grid-even">
      <div className="nf-evidence-bay">
        <p className="nf-bay-label">{liveCurve ? 'training now / live' : 'learning curve / active checkpoint'}</p>
        <h3>{liveCurve ? `Epoch ${curve[curve.length - 1].epoch}` : 'How the champion learned'}</h3>
        {!liveCurve && (
          <div className="stat-line">
            <div><b>{pct(test.accuracy)}</b><span>test accuracy</span></div>
            <div><b>{pct(test.f1)}</b><span>test macro F1</span></div>
            <div><b>{pct(valid.f1)}</b><span>validation F1</span></div>
            <div><b>{test.sample_count ?? '—'}</b><span>test samples</span></div>
          </div>
        )}
        {curve?.length ? <LearningCurve curve={curve} bestEpoch={bestEpoch} /> : <p className="muted small">This checkpoint predates learning-curve logging. Retrain it to record one.</p>}
        {active && !active.test_is_independent && !liveCurve && (
          <div className="notice" style={{ marginTop: 12 }}>Legacy checkpoint: its test split was also used to pick it. Retrain for an independent score.</div>
        )}
      </div>
      {!liveCurve && test.confusion_matrix && (
        <div className="nf-evidence-bay">
          <p className="nf-bay-label">confusion heatmap / test split</p>
          <h3>Where it goes wrong</h3>
          <p className="muted small" style={{ marginTop: 0 }}>Rows are the true class, columns the prediction. Shading is the share of each row; the diagonal is correct.</p>
          <ConfusionMatrix matrix={test.confusion_matrix} classNames={active.class_names || []} />
        </div>
      )}
    </div>
  );
}
