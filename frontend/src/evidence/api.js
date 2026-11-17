export async function api(path, options = {}) {
  const init = { ...options };
  if (options.json !== undefined) {
    init.method = init.method || 'POST';
    init.headers = { 'Content-Type': 'application/json', ...(init.headers || {}) };
    init.body = JSON.stringify(options.json);
    delete init.json;
  }
  const response = await fetch(path, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail || payload);
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return payload;
}

export function searchSocket() {
  const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return new WebSocket(`${scheme}://${window.location.host}/ws/search`);
}

export const pct = (value, digits = 1) =>
  value === null || value === undefined || Number.isNaN(Number(value)) ? '—' : `${Number(value).toFixed(digits)}%`;

export const compact = (value) => {
  if (value === null || value === undefined) return '—';
  const n = Number(value);
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return String(n);
};

export const datasetName = (meta) => {
  const root = meta?.dataset_root || '';
  return root.split('/').filter(Boolean).pop() || 'dataset';
};

/** Human description of one mutation operation. */
export function describeOperation(op) {
  if (op.field === 'elite') return ['Elite', 'copied unchanged'];
  const label = op.field
    .replace(/^filters\[(\d+)\]$/, (_, i) => `Layer ${Number(i) + 1} filters`)
    .replace(/^kernel_sizes\[(\d+)\]$/, (_, i) => `Layer ${Number(i) + 1} kernel`)
    .replace('num_layers', 'Depth')
    .replace('block_type', 'Block type')
    .replace('input_size', 'Input size')
    .replace('batch_norm', 'BatchNorm')
    .replace('activation', 'Activation')
    .replace('dropout', 'Dropout')
    .replace('pooling', 'Pooling');
  const change = 'from' in op ? `${op.from} → ${op.to}` : `now ${op.to ? 'on' : 'off'}`;
  return [label, change];
}

export function describeConfig(config) {
  if (!config) return '';
  const block = config.block_type && config.block_type !== 'plain' ? `${config.block_type} ` : '';
  return `${config.num_layers}-layer ${block}CNN, ${config.activation}, ${config.input_size || 64}px input`;
}
