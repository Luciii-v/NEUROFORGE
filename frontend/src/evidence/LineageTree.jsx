import { useMemo } from 'react';
import { pct } from './api';

const COL_W = 168;
const ROW_H = 50;
const CHIP_W = 124;
const CHIP_H = 36;
const PAD_X = 28;
const PAD_Y = 44;

/** A candidate drawn as its own layer stack: one bar per conv layer, height = log2(filters). */
function LayerGlyph({ config, x, y, color }) {
  const filters = config?.filters || [];
  const block = config?.block_type || 'plain';
  return (
    <g transform={`translate(${x},${y})`}>
      {filters.map((f, i) => {
        const h = 6 + (Math.log2(f) - 4) * 4.5; // 16 -> 6px, 128 -> 19.5px
        return (
          <rect key={i} x={i * 7} y={(22 - h) / 2} width={5} height={h} rx={1} style={{ fill: color }}
            opacity={block === 'depthwise' ? 0.65 : 1} />
        );
      })}
      {block === 'residual' && filters.length > 1 && (
        <path d={`M2 -1 Q ${(filters.length - 1) * 3.5 + 2} -7 ${(filters.length - 1) * 7 + 2} -1`} fill="none" style={{ stroke: color }} strokeWidth="1.2" />
      )}
    </g>
  );
}

export default function LineageTree({ history, winnerId, selectedId, onSelect, theme }) {
  const layout = useMemo(() => {
    const byGen = new Map();
    history.forEach((c) => {
      if (!byGen.has(c.generation)) byGen.set(c.generation, []);
      byGen.get(c.generation).push(c);
    });
    const gens = [...byGen.keys()].sort((a, b) => a - b);
    const positions = new Map();
    let maxRows = 1;
    gens.forEach((g, gi) => {
      const members = byGen.get(g).slice().sort((a, b) => b.score - a.score);
      maxRows = Math.max(maxRows, members.length);
      members.forEach((c, ri) => positions.set(c.candidate_id, { x: PAD_X + (gi + 1) * COL_W, y: PAD_Y + ri * ROW_H, c }));
    });
    const f1s = history.map((c) => c.f1);
    const lo = Math.min(...f1s);
    const hi = Math.max(...f1s);

    // Ancestry of the winner, traced in sodium.
    const winnerPath = new Set();
    const byId = new Map(history.map((c) => [c.candidate_id, c]));
    let cursor = byId.get(winnerId);
    while (cursor) {
      winnerPath.add(cursor.candidate_id);
      cursor = byId.get(cursor.parent_id);
    }
    return {
      gens,
      positions,
      width: PAD_X * 2 + (gens.length + 1) * COL_W,
      height: PAD_Y + maxRows * ROW_H + 8,
      lo,
      hi: hi > lo ? hi : lo + 1,
      winnerPath,
    };
  }, [history, winnerId]);

  // Colours come from the enclosing panel's CSS variables so the tree matches its bay.
  const ink = 'var(--ink)';
  const ink3 = 'var(--ink-3)';
  const line = 'var(--line-strong)';
  const sodium = 'var(--sodium)';
  const panel = 'var(--panel)';
  void theme;

  const seed = { x: PAD_X, y: PAD_Y };

  return (
    <svg width={layout.width} height={layout.height} role="img"
      aria-label={`Lineage of ${history.length} evaluated candidates across ${layout.gens.length} generations`}>
      {layout.gens.map((g, gi) => (
        <text key={g} x={PAD_X + (gi + 1) * COL_W} y={22} style={{ fill: ink3 }} fontSize="12.5">Generation {g}</text>
      ))}
      <text x={PAD_X} y={22} style={{ fill: ink3 }} fontSize="12.5">Seed</text>

      {/* edges */}
      {history.map((c) => {
        const to = layout.positions.get(c.candidate_id);
        const from = c.parent_id ? layout.positions.get(c.parent_id) : seed;
        if (!to || !from) return null;
        const x1 = from.x + (c.parent_id ? CHIP_W : 26);
        const y1 = from.y + CHIP_H / 2;
        const x2 = to.x;
        const y2 = to.y + CHIP_H / 2;
        const onPath = layout.winnerPath.has(c.candidate_id) && (!c.parent_id || layout.winnerPath.has(c.parent_id));
        const elite = c.mutation_operations?.some((op) => op.field === 'elite');
        const mid = (x1 + x2) / 2;
        return (
          <path key={`e-${c.candidate_id}`} d={`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`}
            fill="none" style={{ stroke: onPath ? sodium : line }} strokeWidth={onPath ? 2.2 : 1}
            strokeDasharray={elite ? '4 3' : undefined} opacity={onPath ? 1 : 0.8} />
        );
      })}

      {/* seed */}
      <g transform={`translate(${seed.x},${seed.y})`}>
        <rect width={26} height={CHIP_H} rx={3} fill="none" style={{ stroke: line }} />
        <circle cx={13} cy={CHIP_H / 2} r={3.5} style={{ fill: ink3 }} />
      </g>

      {/* nodes */}
      {[...layout.positions.values()].map(({ x, y, c }) => {
        const t = (c.f1 - layout.lo) / (layout.hi - layout.lo);
        const fill = `color-mix(in srgb, var(--heat-hi) ${Math.round((0.12 + t * 0.8) * 100)}%, var(--heat-lo))`;
        const textOnFill = t > 0.55 ? panel : ink;
        const isWinner = c.candidate_id === winnerId;
        const selected = c.candidate_id === selectedId;
        return (
          <g key={c.candidate_id} className="node" transform={`translate(${x},${y})`} tabIndex={0} role="button"
            aria-label={`${c.candidate_id}: validation F1 ${pct(c.f1)}${isWinner ? ', winner' : ''}`}
            aria-pressed={selected}
            onClick={() => onSelect(c.candidate_id)}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onSelect(c.candidate_id))}>
            <title>{`${c.candidate_id}\nvalidation F1 ${pct(c.f1)}, accuracy ${pct(c.accuracy)}\nscore ${c.score.toFixed(1)}${c.evaluation_reused ? '\n(evaluation reused from cache)' : ''}`}</title>
            <rect className="chip" width={CHIP_W} height={CHIP_H} rx={3}
              style={{ fill, stroke: isWinner ? sodium : selected ? ink : 'none' }} strokeWidth={isWinner || selected ? 2.5 : 0}
              strokeDasharray={c.evaluation_reused ? '3 2' : undefined} />
            <LayerGlyph config={c.config} x={9} y={7} color={textOnFill} />
            <text x={CHIP_W - 9} y={CHIP_H / 2 + 5} textAnchor="end" style={{ fill: textOnFill }} fontSize="13.5" fontWeight="650">
              {c.f1.toFixed(1)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
