/** Technical drawing of a DynamicCNN config: input -> conv blocks -> pooling -> linear head. */
export default function ArchitectureDrawing({ config, numClasses, height = 132 }) {
  if (!config?.filters) return null;
  const layers = config.filters.map((f, i) => ({ filters: f, kernel: config.kernel_sizes[i] }));
  const size = config.input_size || 64;
  const block = config.block_type || 'plain';
  const unit = 24;
  const gap = 18;
  const blockW = (k) => (k === 5 ? 30 : 22);
  const blockH = (f) => 26 + (Math.log2(f) - 4) * 18;
  const widths = [unit, ...layers.map((l) => blockW(l.kernel)), 18, 22];
  const total = widths.reduce((a, b) => a + b, 0) + gap * (widths.length - 1);
  const mid = height / 2 - 6;
  let x = 0;
  const parts = [];

  parts.push(
    <g key="in">
      <rect x={x} y={mid - 22} width={unit} height={44} fill="none" stroke="var(--ink-2)" strokeWidth="1.2" />
      <text x={x + unit / 2} y={height - 4} textAnchor="middle" fontSize="11" fill="var(--ink-3)">{size}px</text>
    </g>,
  );
  x += unit + gap;

  layers.forEach((layer, i) => {
    const w = blockW(layer.kernel);
    const h = blockH(layer.filters);
    parts.push(
      <g key={`l${i}`}>
        <line x1={x - gap} y1={mid} x2={x} y2={mid} stroke="var(--ink-3)" />
        <rect x={x} y={mid - h / 2} width={w} height={h} rx={2} fill="var(--panel-raised)" stroke="var(--ink)" strokeWidth="1.2"
          strokeDasharray={block === 'depthwise' ? '3 2' : undefined} />
        {block === 'residual' && (
          <path d={`M${x - 6},${mid - h / 2 - 4} q ${w / 2 + 6} -14 ${w + 12} 0`} fill="none" stroke="var(--ink-2)" strokeWidth="1" />
        )}
        <text x={x + w / 2} y={mid + 4} textAnchor="middle" fontSize="11" fill="var(--ink)" fontWeight="600">{layer.filters}</text>
        <text x={x + w / 2} y={height - 4} textAnchor="middle" fontSize="11" fill="var(--ink-3)">{layer.kernel}×{layer.kernel}</text>
      </g>,
    );
    x += w + gap;
  });

  parts.push(
    <g key="gap">
      <line x1={x - gap} y1={mid} x2={x} y2={mid} stroke="var(--ink-3)" />
      <circle cx={x + 9} cy={mid} r={8} fill="none" stroke="var(--ink-2)" />
      <text x={x + 9} y={height - 4} textAnchor="middle" fontSize="11" fill="var(--ink-3)">pool</text>
    </g>,
  );
  x += 18 + gap;
  parts.push(
    <g key="head">
      <line x1={x - gap} y1={mid} x2={x} y2={mid} stroke="var(--ink-3)" />
      <rect x={x} y={mid - 16} width={22} height={32} rx={2} fill="var(--sodium)" />
      <text x={x + 11} y={height - 4} textAnchor="middle" fontSize="11" fill="var(--ink-3)">{numClasses || 'N'}</text>
    </g>,
  );

  return (
    <svg viewBox={`-4 0 ${total + 8} ${height}`} width="100%" height={height} role="img"
      aria-label={`${layers.length} convolution layers with ${layers.map((l) => l.filters).join(', ')} filters, then global pooling and a ${numClasses || 'N'}-class head`}>
      {parts}
    </svg>
  );
}
