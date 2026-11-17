import { useRef, useState } from 'react';
import { FolderOpen, Upload, X } from 'lucide-react';
import { api, datasetName } from '../api';

export default function DataSection({ dataset, mounted, busy, onChange }) {
  const [path, setPath] = useState('');
  const [error, setError] = useState('');
  const [working, setWorking] = useState('');
  const [over, setOver] = useState(false);
  const fileRef = useRef(null);

  const run = async (label, action) => {
    setWorking(label);
    setError('');
    try {
      await action();
      await onChange();
    } catch (e) {
      setError(e.message);
    } finally {
      setWorking('');
    }
  };

  const mount = (event) => {
    event.preventDefault();
    if (!path.trim()) return;
    run('Inspecting dataset…', () => api('/api/dataset/load', { json: { path: path.trim() } }));
  };

  const upload = (file) => {
    if (!file) return;
    const body = new FormData();
    body.append('file', file);
    run(`Extracting ${file.name}…`, () => api('/api/dataset/upload-zip', { method: 'POST', body }));
  };

  const counts = dataset?.class_counts ? Object.entries(dataset.class_counts) : [];
  const maxCount = Math.max(1, ...counts.map(([, n]) => n));
  const splits = dataset?.splits;
  const imbalance = counts.length ? Math.max(...counts.map(([, n]) => n)) / Math.max(1, Math.min(...counts.map(([, n]) => n))) : 1;

  return (
    <section id="step-data" className="section" aria-labelledby="data-title">
      <div className="section-head">
        <h2 id="data-title">Data</h2>
        <p>Mount a folder of class sub-folders (images or WAV), a Roboflow classification export, or ESC-50 / UrbanSound8K.</p>
      </div>
      <div className="data-grid">
        <div className="panel panel-pad">
          <form onSubmit={mount} className="field" style={{ gap: 8 }}>
            <label htmlFor="ds-path">Dataset folder on this computer</label>
            <input id="ds-path" value={path} onChange={(e) => setPath(e.target.value)} placeholder="/Users/you/datasets/crop-pests" autoComplete="off" />
            <button className="btn" type="submit" disabled={busy || !!working || !path.trim()}>
              <FolderOpen size={16} aria-hidden="true" />Mount folder
            </button>
          </form>
          <p className="muted small" style={{ margin: '14px 0 8px' }}>Or upload a ZIP (up to 2 GB)</p>
          <div className={`dropzone${over ? ' over' : ''}`} role="button" tabIndex={0}
            onClick={() => fileRef.current?.click()}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && fileRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); upload(e.dataTransfer.files[0]); }}>
            <Upload size={18} aria-hidden="true" /> Drop a .zip here or choose one
            <input ref={fileRef} type="file" accept=".zip" hidden onChange={(e) => upload(e.target.files[0])} />
          </div>
          {working && <p className="small" style={{ marginBottom: 0 }}>{working}</p>}
          {error && <div className="notice error" style={{ marginTop: 12 }} role="alert">{error}</div>}
          {mounted && (
            <button type="button" className="btn small" style={{ marginTop: 14 }} disabled={busy}
              onClick={() => run('Unmounting…', () => api('/api/dataset/unload', { json: { delete_uploaded: false } }))}>
              <X size={14} aria-hidden="true" />Unmount
            </button>
          )}
        </div>

        <div className="panel panel-pad">
          {dataset ? (
            <>
              <h3 title={dataset.dataset_root}>{datasetName(dataset)}</h3>
              <div className="stat-line">
                <div><b>{dataset.num_classes}</b><span>classes</span></div>
                <div><b>{dataset.num_samples ?? '—'}</b><span>{dataset.modality === 'audio' ? 'clips' : 'images'}</span></div>
                {splits && <div><b>{splits.train} / {splits.valid} / {splits.test}</b><span>train / valid / test</span></div>}
              </div>
              {counts.length > 0 && (
                <>
                  <div className="hbars" role="table" aria-label="Images per class">
                    {counts.map(([name, n]) => (
                      <div key={name} role="row" style={{ display: 'contents' }}>
                        <span className="lab" role="cell" title={name}>{name}</span>
                        <span className="track" role="cell"><span className="bar" style={{ width: `${(n / maxCount) * 100}%`, display: 'block' }} /></span>
                        <span className="val" role="cell">{n}</span>
                      </div>
                    ))}
                  </div>
                  {imbalance >= 2 && (
                    <p className="muted small" style={{ marginBottom: 0 }}>
                      The largest class is {imbalance.toFixed(1)}× the smallest. Training uses class-weighted loss, and results are reported as macro F1 so small classes count equally.
                    </p>
                  )}
                </>
              )}
              {splits && splits.test < 60 && (
                <div className="notice warn" style={{ marginTop: 12 }}>
                  Only {splits.test} test samples: one image moves test accuracy by {(100 / splits.test).toFixed(1)} points. Treat small differences between models as noise.
                </div>
              )}
            </>
          ) : (
            <div>
              <h3>No dataset mounted</h3>
              <p className="muted" style={{ margin: 0 }}>Mount a folder or drop a ZIP to begin. NEUROFORGE splits it 70 / 15 / 15 into train, validation and a sealed test set, stratified by class.</p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
