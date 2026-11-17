import { useEffect, useRef, useState } from 'react';
import { ImageUp } from 'lucide-react';
import { api, pct } from '../api';

export default function TestBench({ ready, modelKey }) {
  const [samples, setSamples] = useState([]);
  const [preview, setPreview] = useState(null); // { url, kind, name }
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [working, setWorking] = useState(false);
  const [camOpacity, setCamOpacity] = useState(0.6);
  const [expected, setExpected] = useState(null);
  const fileRef = useRef(null);

  useEffect(() => {
    api('/api/samples').then((p) => setSamples(p.samples || [])).catch(() => setSamples([]));
  }, [modelKey]);

  const classify = async (blob, name, truth = null) => {
    const kind = name.toLowerCase().endsWith('.wav') ? 'audio' : 'image';
    if (preview?.url) URL.revokeObjectURL(preview.url);
    setPreview({ url: URL.createObjectURL(blob), kind, name });
    setExpected(truth);
    setResult(null);
    setError('');
    setWorking(true);
    try {
      const body = new FormData();
      body.append('file', blob, name);
      setResult(await api('/api/predict', { method: 'POST', body }));
    } catch (e) {
      setError(e.message);
    } finally {
      setWorking(false);
    }
  };

  const runSample = async (sample) => {
    const endpoint = sample.modality === 'audio' ? '/api/sample-audio' : '/api/sample-image';
    const response = await fetch(`${endpoint}?rel_path=${encodeURIComponent(sample.rel_path)}`);
    if (!response.ok) return setError('Could not load that sample.');
    classify(await response.blob(), sample.filename, sample.class_name);
  };

  const ranked = result?.all_classes?.slice(0, 6) || [];
  const correct = expected && result ? result.prediction === expected : null;

  return (
    <section id="step-test" className="section" aria-labelledby="test-title">
      <div className="section-head">
        <h2 id="test-title">Test</h2>
        <p>Try the active model on a held-out sample or your own file. The heat overlay (Grad-CAM) shows which pixels pushed the model toward its answer.</p>
      </div>
      {!ready ? (
        <div className="panel panel-pad"><p className="muted" style={{ margin: 0 }}>Train a model first; the test bench uses whichever model is active.</p></div>
      ) : (
        <div className="bench-grid">
          <div className="panel panel-pad">
            <button type="button" className="btn" onClick={() => fileRef.current?.click()} disabled={working}>
              <ImageUp size={16} aria-hidden="true" />Choose an image or WAV
            </button>
            <input ref={fileRef} type="file" accept="image/*,.wav" hidden onChange={(e) => e.target.files[0] && classify(e.target.files[0], e.target.files[0].name)} />
            {samples.length > 0 && (
              <>
                <p className="muted small" style={{ margin: '14px 0 0' }}>One sample per class from the sealed test split</p>
                <div className="samples">
                  {samples.map((s) => (
                    <button type="button" key={s.rel_path} className="sample" onClick={() => runSample(s)} disabled={working} title={s.class_name}>
                      {s.modality === 'image'
                        ? <img src={`/api/sample-image?rel_path=${encodeURIComponent(s.rel_path)}`} alt="" loading="lazy" />
                        : <span className="tag">WAV</span>}
                      <span>{s.class_name}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
            {preview && (
              <div style={{ marginTop: 16 }}>
                {preview.kind === 'image' ? (
                  <div className="specimen">
                    <img src={preview.url} alt={`Input: ${preview.name}`} />
                    {result?.saliency_map && <img className="cam" src={result.saliency_map} alt="" style={{ opacity: camOpacity }} />}
                  </div>
                ) : (
                  <audio controls src={preview.url} style={{ width: '100%' }} />
                )}
                {result?.saliency_map && (
                  <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: 10, marginTop: 10 }}>
                    Heat overlay
                    <input type="range" min="0" max="1" step="0.05" value={camOpacity} onChange={(e) => setCamOpacity(Number(e.target.value))} style={{ flex: 1, padding: 0, border: 0 }} />
                  </label>
                )}
              </div>
            )}
          </div>

          <div className="panel panel-pad" aria-live="polite">
            {working && <p className="muted">Classifying…</p>}
            {error && <div className="notice error" role="alert">{error}</div>}
            {!working && !result && !error && <p className="muted" style={{ margin: 0 }}>Pick a sample to see the prediction, its confidence against chance, and how long it took.</p>}
            {result && (
              <>
                <p className="muted small" style={{ margin: 0 }}>Prediction</p>
                <p className="verdict">{result.prediction}</p>
                <p className="small" style={{ margin: 0, color: 'var(--ink-2)' }}>
                  {pct(result.percentage)} confident, {result.latency_ms} ms
                  {correct !== null && (correct
                    ? <span style={{ color: 'var(--good)' }}>. Correct (labelled {expected})</span>
                    : <span style={{ color: 'var(--rust)' }}>. Wrong: labelled {expected}</span>)}
                </p>
                <div className="pred-bars" role="table" aria-label="Class probabilities">
                  {ranked.map((c, i) => (
                    <div key={c.class_index} role="row" style={{ display: 'contents' }}>
                      <span role="cell" title={c.class_name} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.class_name}</span>
                      <span role="cell" className="track">
                        <span className={`bar${i === 0 ? ' top' : ''}`} style={{ width: `${c.percentage}%`, display: 'block' }} />
                        <span className="chance" style={{ left: `${result.chance_percentage}%` }} aria-hidden="true" />
                      </span>
                      <span role="cell" className="num" style={{ textAlign: 'right' }}>{pct(c.percentage)}</span>
                    </div>
                  ))}
                </div>
                <p className="muted small" style={{ marginBottom: 0 }}>Dashed line: chance level ({pct(result.chance_percentage)}). Confidence is not accuracy; check the test scores in Compare.</p>
              </>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
