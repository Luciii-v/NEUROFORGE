import { useRef, useState } from 'react';
import { pct } from '../api';

/** Train accuracy vs validation F1 per epoch (same unit, one axis). Marks the selected epoch. */
export function LearningCurve({ curve, bestEpoch, height = 220 }) {
  const ref = useRef(null);
  const [hover, setHover] = useState(null);
  if (!curve?.length) return null;
  const width = 560;
  const m = { l: 36, r: 64, t: 10, b: 26 };
  const w = width - m.l - m.r;
  const h = height - m.t - m.b;
  const n = curve.length;
  const x = (i) => m.l + (n === 1 ? w / 2 : (i / (n - 1)) * w);
  const y = (v) => m.t + h - (Math.max(0, Math.min(100, v)) / 100) * h;
  const series = [
    { key: 'valid_f1', label: 'Validation F1', color: 'var(--series-a)' },
    { key: 'train_accuracy', label: 'Train accuracy', color: 'var(--series-b)' },
  ];
  const path = (key) => curve.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`).join('');
  const bestIndex = curve.findIndex((p) => p.epoch === bestEpoch);

  const onMove = (event) => {
    const box = ref.current.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * width;
    const i = Math.round(((px - m.l) / w) * (n - 1));
    if (i >= 0 && i < n) setHover({ i, left: (x(i) / width) * box.width });
    else setHover(null);
  };

  return (
    <div>
      <div className="legend" aria-hidden="true">
        {series.map((s) => <span key={s.key}><i style={{ background: s.color }} />{s.label}</span>)}
        {bestEpoch ? <span><i style={{ background: 'var(--sodium)', width: 3, height: 12 }} />Kept checkpoint (epoch {bestEpoch})</span> : null}
      </div>
      <div style={{ position: 'relative' }} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        <svg ref={ref} viewBox={`0 0 ${width} ${height}`} width="100%" role="img"
          aria-label={`Learning curve over ${n} epochs. Checkpoint kept at epoch ${bestEpoch}.`}>
          {[0, 25, 50, 75, 100].map((v) => (
            <g key={v}>
              <line x1={m.l} x2={m.l + w} y1={y(v)} y2={y(v)} stroke="var(--line)" />
              <text x={m.l - 6} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--ink-3)">{v}</text>
            </g>
          ))}
          <text x={m.l} y={height - 6} fontSize="11" fill="var(--ink-3)">epoch 1</text>
          <text x={m.l + w} y={height - 6} fontSize="11" fill="var(--ink-3)" textAnchor="end">{curve[n - 1].epoch}</text>
          {bestIndex >= 0 && <line x1={x(bestIndex)} x2={x(bestIndex)} y1={m.t} y2={m.t + h} stroke="var(--sodium)" strokeWidth="2" />}
          {series.map((s) => (
            <g key={s.key}>
              <path d={path(s.key)} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" />
              <text x={x(n - 1) + 6} y={y(curve[n - 1][s.key]) + 4} fontSize="11.5" fill="var(--ink-2)">{s.key === 'valid_f1' ? 'val F1' : 'train'}</text>
            </g>
          ))}
          {hover && (
            <g>
              <line x1={x(hover.i)} x2={x(hover.i)} y1={m.t} y2={m.t + h} stroke="var(--ink-3)" strokeDasharray="3 3" />
              {series.map((s) => (
                <circle key={s.key} cx={x(hover.i)} cy={y(curve[hover.i][s.key])} r={4.5} fill={s.color} stroke="var(--panel)" strokeWidth="2" />
              ))}
            </g>
          )}
        </svg>
        {hover && (
          <div className="chart-tip" style={{ left: Math.min(hover.left + 12, 380), top: 8 }}>
            <strong>Epoch {curve[hover.i].epoch}</strong><br />
            Validation F1 {pct(curve[hover.i].valid_f1)}<br />
            Train accuracy {pct(curve[hover.i].train_accuracy)}
          </div>
        )}
      </div>
      <details className="table-view">
        <summary>Show as table</summary>
        <table className="mini-table">
          <thead><tr><th>Epoch</th><th>Train loss</th><th>Train acc</th><th>Valid acc</th><th>Valid F1</th></tr></thead>
          <tbody>
            {curve.map((p) => (
              <tr key={p.epoch} style={p.epoch === bestEpoch ? { fontWeight: 700 } : undefined}>
                <td>{p.epoch}</td><td>{p.train_loss?.toFixed(3)}</td><td>{pct(p.train_accuracy)}</td><td>{pct(p.valid_accuracy)}</td><td>{pct(p.valid_f1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

/** Rows = actual class, columns = predicted. Shading is row-normalised (per-class recall). */
export function ConfusionMatrix({ matrix, classNames }) {
  if (!matrix?.length) return null;
  const short = (name) => (name.length > 14 ? `${name.slice(0, 13)}…` : name);
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="confusion">
        <caption className="sr-only">Confusion matrix on the test split. Rows are the true class, columns the predicted class.</caption>
        <thead>
          <tr>
            <th scope="col"><span className="sr-only">True class</span></th>
            {classNames.map((c, j) => <th key={j} scope="col" title={c}>{short(c)}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={i}>
                <th scope="row" className="rowh" title={classNames[i]}>{short(classNames[i] || `Class ${i}`)}</th>
                {row.map((count, j) => {
                  const share = count / total;
                  return (
                    <td key={j}
                      title={`${classNames[i]} predicted as ${classNames[j]}: ${count} of ${total} (${Math.round(share * 100)}%)`}
                      style={{
                        background: `color-mix(in srgb, var(--heat-hi) ${Math.round(share * 85)}%, var(--heat-lo))`,
                        color: share > 0.5 ? 'var(--panel)' : 'var(--ink)',
                        outline: i === j ? '1.5px solid var(--line-strong)' : undefined,
                      }}>
                      {count}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
