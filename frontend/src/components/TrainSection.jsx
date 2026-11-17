import { AlertTriangle } from 'lucide-react';
import { compact, describeConfig, pct } from '../api';
import ArchitectureDrawing from './ArchitectureDrawing';
import { ConfusionMatrix, LearningCurve } from './Charts';

export default function TrainSection({ active, job, matchesDataset, trainError }) {
  const training = job?.running && job.kind === 'train';
  const liveCurve = training
    ? job.progress.filter((p) => p.epoch).map((p) => ({ epoch: p.epoch, train_accuracy: p.train_acc, valid_f1: p.val_f1, valid_accuracy: p.val_acc }))
    : null;
  const latest = liveCurve?.[liveCurve.length - 1];
  const best = liveCurve?.length ? liveCurve.reduce((a, b) => (b.valid_f1 > a.valid_f1 ? b : a)) : null;
  const test = active?.test_metrics || {};
  const valid = active?.validation_metrics || {};

  return (
    <section id="step-train" className="section" aria-labelledby="train-title">
      <div className="section-head">
        <h2 id="train-title">Train</h2>
        <p>The chosen design trains from fresh weights. Validation picks the checkpoint and stops training when it stops improving. The sealed test split is opened once, at the end.</p>
      </div>
      {trainError && <div className="notice error" role="alert" style={{ marginBottom: 16 }}>{trainError}</div>}

      {training ? (
        <div className="panel panel-pad">
          <h3>{job.label}</h3>
          {latest ? (
            <>
              <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={job.progress[job.progress.length - 1].total_epochs} aria-valuenow={latest.epoch}>
                <div style={{ width: `${(latest.epoch / job.progress[job.progress.length - 1].total_epochs) * 100}%` }} />
              </div>
              <p className="small muted" aria-live="polite">
                Epoch {latest.epoch} of up to {job.progress[job.progress.length - 1].total_epochs}. Best validation F1 so far {pct(best.valid_f1)} at epoch {best.epoch}. Training stops early if validation stops improving.
              </p>
              <LearningCurve curve={liveCurve} bestEpoch={best.epoch} />
            </>
          ) : <p className="muted">Loading data and building the model…</p>}
        </div>
      ) : active?.ready ? (
        <div className="train-grid">
          <div className="panel panel-pad">
            <div className="row" style={{ justifyContent: 'space-between', alignItems: 'baseline' }}>
              <h3 style={{ margin: 0 }}>Active model</h3>
              <span className="muted small">{active.bundle_id || 'legacy checkpoint'}</span>
            </div>
            <p className="muted small" style={{ margin: '4px 0 12px' }}>{describeConfig(active.config)}, {compact(active.parameters)} parameters</p>
            {!active.test_is_independent && (
              <div className="notice warn" style={{ marginBottom: 12 }}>
                <AlertTriangle size={16} aria-hidden="true" />
                <span>This checkpoint was trained before the train/valid/test fix, so its test score was also used to pick it. Retrain it for an independent number.</span>
              </div>
            )}
            {!matchesDataset && (
              <div className="notice warn" style={{ marginBottom: 12 }}>
                <AlertTriangle size={16} aria-hidden="true" />
                <span>The active model was trained on a different dataset from the one mounted now.</span>
              </div>
            )}
            <div className="stat-line">
              <div><b>{pct(test.accuracy)}</b><span>test accuracy</span></div>
              <div><b>{pct(test.f1)}</b><span>test macro F1</span></div>
              <div><b>{pct(valid.f1)}</b><span>validation F1</span></div>
              <div><b>{test.sample_count ?? '—'}</b><span>test samples</span></div>
            </div>
            {active.learning_curve?.length ? (
              <LearningCurve curve={active.learning_curve} bestEpoch={active.best_epoch} />
            ) : <ArchitectureDrawing config={active.config} numClasses={active.class_names?.length} />}
          </div>
          <div className="panel panel-pad">
            <h3>Where it goes wrong</h3>
            <p className="muted small" style={{ marginTop: 0 }}>Test split. Rows are the true class and columns the prediction; shading is the share of each row.</p>
            <ConfusionMatrix matrix={test.confusion_matrix} classNames={active.class_names || []} />
          </div>
        </div>
      ) : (
        <div className="panel panel-pad">
          <h3>No trained model yet</h3>
          <p className="muted" style={{ margin: 0 }}>Run a search, then choose <strong>Train this design</strong> in the inspector. The search winner is selected for you.</p>
        </div>
      )}
    </section>
  );
}
