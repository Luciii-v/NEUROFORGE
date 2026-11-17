import { useEffect, useState } from 'react';
import { Check, Copy, Download, FileJson, Package } from 'lucide-react';
import { api, pct } from '../api';

function snippet(active) {
  const pre = active?.preprocessing || {};
  const size = Array.isArray(pre.resize) ? pre.resize[0] : 64;
  return `import json, torch
from PIL import Image
import torchvision.transforms as T
from neuroforge_core import DynamicCNN

bundle = "path/to/unzipped-bundle"
config = json.load(open(f"{bundle}/config.json"))
labels = json.load(open(f"{bundle}/labels.json"))

model = DynamicCNN(config, num_classes=len(labels))
model.load_state_dict(torch.load(f"{bundle}/weights.pth", map_location="cpu"))
model.eval()

prep = T.Compose([
    T.Resize((${size}, ${size})),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
x = prep(Image.open("leaf.jpg").convert("RGB")).unsqueeze(0)
probs = model(x).softmax(dim=1)[0]
print(labels[int(probs.argmax())], float(probs.max()))`;
}

const MODE_NAMES = {
  dynamic_cnn: 'manual CNN',
  transfer_learning_resnet18: 'ResNet-18',
  transfer_learning_efficientnet_b0: 'EfficientNet-B0',
  transfer_learning_mobilenet_v3: 'MobileNet-V3',
  transfer_learning_shufflenet_v2: 'ShuffleNet-V2',
};
const modeName = (mode) => MODE_NAMES[mode] || mode || 'model';

function runDate(runId) {
  const m = runId.match(/^(\d{4})(\d\d)(\d\d)T(\d\d)(\d\d)/);
  if (!m) return runId;
  const date = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]));
  return date.toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
}

export default function ShipSection({ active, refreshKey }) {
  const [runs, setRuns] = useState([]);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    api('/api/runs').then((p) => setRuns(p.runs || [])).catch(() => setRuns([]));
  }, [refreshKey]);

  const copy = async () => {
    await navigator.clipboard.writeText(snippet(active));
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <section id="step-ship" className="section" aria-labelledby="ship-title">
      <div className="section-head">
        <h2 id="ship-title">Ship</h2>
        <p>Download the active model as a self-contained bundle: weights plus the config, class labels, preprocessing, split manifest, metrics and lineage needed to reproduce and run it.</p>
      </div>
      <div className="ship-grid">
        <div className="panel panel-pad">
          <div className="downloads">
            {active?.bundle_id ? (
              <a href="/api/download/bundle" download>
                <Package size={18} aria-hidden="true" /><strong>Model bundle (.zip)</strong><span>{active.bundle_id.split('/').pop()}</span>
              </a>
            ) : (
              <p className="muted small" style={{ margin: 0 }}>This checkpoint predates bundles. Retrain it to get a full bundle.</p>
            )}
            <a href="/api/download/weights" download><Download size={18} aria-hidden="true" /><strong>Weights only</strong><span>.pth state dict</span></a>
            <a href="/api/download/config" download><FileJson size={18} aria-hidden="true" /><strong>Architecture config</strong><span>.json</span></a>
          </div>
          <div className="row" style={{ justifyContent: 'space-between', margin: '18px 0 8px' }}>
            <h3 style={{ margin: 0 }}>Run it in PyTorch</h3>
            <button type="button" className="btn small" onClick={copy} disabled={!active?.ready}>
              {copied ? <><Check size={14} aria-hidden="true" />Copied</> : <><Copy size={14} aria-hidden="true" />Copy</>}
            </button>
          </div>
          <pre className="code"><code>{snippet(active)}</code></pre>
        </div>
        <div className="panel panel-pad">
          <h3>Run history</h3>
          {runs.length ? (
            <ul className="runs">
              {runs.slice(0, 12).map((run) => {
                const trained = run.bundles.filter((b) => b.test_f1 !== undefined && b.test_f1 !== null);
                return (
                  <li key={run.run_id}>
                    <span><strong>{run.kind === 'search' ? 'Search' : `Baseline: ${modeName(run.bundles[0]?.training_mode)}`}</strong></span>
                    <span className="muted small">{runDate(run.run_id)}</span>
                    <span className="muted small" style={{ gridColumn: '1 / -1' }}>
                      {run.kind === 'search' && `${run.candidates} designs, best ${pct(run.best_metrics?.f1)} validation F1`}
                      {trained.map((b, i) => `${run.kind === 'search' || i ? '; ' : ''}${run.kind === 'search' ? 'trained to ' : ''}${pct(b.test_f1)} test F1`).join('')}
                    </span>
                  </li>
                );
              })}
            </ul>
          ) : <p className="muted" style={{ margin: 0 }}>Each search and training run is stored under runs/ and listed here.</p>}
        </div>
      </div>
    </section>
  );
}
