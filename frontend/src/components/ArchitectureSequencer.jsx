import React, { useState } from 'react';
import { Dna, Plus, Trash2, Zap, Play, RotateCcw, Check, Sparkles, Layers, Sliders } from 'lucide-react';
import soundEngine from '../utils/soundEngine';

export default function ArchitectureSequencer({
  initialConfig = {
    num_layers: 3,
    filters: [32, 64, 128],
    kernel_sizes: [3, 5, 3],
    batch_norm: true,
    activation: 'GELU',
    pooling: true,
    dropout: 0.0,
  },
  numClasses = 9,
  championAcc = 84.3,
  onApplyConfig = null,
}) {
  const [layers, setLayers] = useState(() => {
    return (initialConfig?.filters || [32, 64, 128]).map((f, i) => ({
      id: i,
      filters: f,
      kernel: initialConfig?.kernel_sizes?.[i] || 3,
    }));
  });

  const [activation, setActivation] = useState(initialConfig?.activation || 'GELU');
  const [batchNorm, setBatchNorm] = useState(initialConfig?.batch_norm ?? true);
  const [dropout, setDropout] = useState(initialConfig?.dropout || 0.0);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);

  // Compute parameter count dynamically
  const calcParams = () => {
    let params = 0;
    let prevChannels = 3;
    layers.forEach((l) => {
      // weights: k*k * in * out + bias
      params += l.kernel * l.kernel * prevChannels * l.filters + l.filters;
      if (batchNorm) params += 4 * l.filters; // bn weights and biases
      prevChannels = l.filters;
    });
    // Classifier head: prevChannels * numClasses + numClasses
    params += prevChannels * numClasses + numClasses;
    return params;
  };

  const handleAddLayer = () => {
    if (layers.length >= 5) return;
    soundEngine.tick(1600, 0.015);
    const lastFilter = layers[layers.length - 1]?.filters || 64;
    const nextFilter = Math.min(256, lastFilter * 2);
    setLayers([...layers, { id: Date.now(), filters: nextFilter, kernel: 3 }]);
  };

  const handleRemoveLayer = (idx) => {
    if (layers.length <= 1) return;
    soundEngine.tick(800, 0.015);
    setLayers(layers.filter((_, i) => i !== idx));
  };

  const handleFilterChange = (idx, newFilter) => {
    soundEngine.tick(1200, 0.01);
    const next = [...layers];
    next[idx].filters = Number(newFilter);
    setLayers(next);
  };

  const handleKernelChange = (idx, newKernel) => {
    soundEngine.tick(1400, 0.01);
    const next = [...layers];
    next[idx].kernel = Number(newKernel);
    setLayers(next);
  };

  const handleSimulateChallenge = () => {
    soundEngine.inferStart();
    setTesting(true);
    setTestResult(null);

    setTimeout(() => {
      const paramCount = calcParams();
      // Heuristic proxy estimate based on depth, GELU, and filter balance
      let score = 65;
      if (activation === 'GELU' || activation === 'Mish') score += 8;
      if (batchNorm) score += 6;
      if (layers.length === 3) score += 6;
      if (layers.some((l) => l.kernel === 5)) score += 3;
      if (dropout > 0.3) score -= 4;
      const finalEst = Math.min(88, Math.max(48, score + (Math.random() * 4 - 2)));

      setTesting(false);
      setTestResult({
        estimatedAccuracy: finalEst.toFixed(1),
        params: paramCount,
        beatChampion: finalEst >= championAcc,
      });

      if (finalEst >= championAcc) {
        soundEngine.championChord(true);
      } else {
        soundEngine.tick(600, 0.02);
      }
    }, 1200);
  };

  const currentParams = calcParams();

  return (
    <div className="w-full rounded-3xl p-6 md:p-8 border border-white/[0.08] relative overflow-hidden backdrop-blur-2xl bg-gradient-to-b from-[#090e18]/90 via-[#05080f]/95 to-[#030508] shadow-2xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 border-b border-white/[0.06] mb-6">
        <div>
          <div className="flex items-center gap-2 font-mono text-[10px] tracking-[0.2em] text-emerald-400 uppercase mb-1">
            <Dna size={13} />
            <span>GENE EDITOR // INTERACTIVE TOPOLOGY SEQUENCER</span>
          </div>
          <h3 className="font-sans text-xl md:text-2xl font-black tracking-tight text-white flex items-center gap-2">
            Human vs. Machine DNA Sequencer
          </h3>
          <p className="font-mono text-xs text-zinc-400 mt-0.5">
            Manually edit convolutional filters, kernels, and non-linearities to challenge the AI Architect.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="font-mono text-xs text-zinc-400 bg-white/[0.03] px-3.5 py-1.5 rounded-full border border-white/[0.06]">
            MODEL PARAMS: <strong className="text-cyan-400 font-bold">{(currentParams / 1000).toFixed(1)}k</strong>
          </div>
        </div>
      </div>

      {/* Layer Sequencer Tracks */}
      <div className="space-y-3 mb-6">
        <div className="flex items-center justify-between font-mono text-[10px] uppercase text-zinc-500 tracking-wider">
          <span>Convolutional Feature Blocks ({layers.length} Layers)</span>
          <span>Kernel / Channel Specs</span>
        </div>

        <div className="grid grid-cols-1 gap-2.5">
          {layers.map((l, idx) => (
            <div
              key={l.id}
              className="p-4 rounded-2xl bg-white/[0.02] border border-white/[0.06] hover:border-white/15 transition-all flex flex-wrap items-center justify-between gap-4 font-mono text-xs"
            >
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded-lg bg-cyan-500/15 text-cyan-400 border border-cyan-400/30 flex items-center justify-center font-bold text-[11px]">
                  L{idx + 1}
                </span>
                <span className="text-zinc-200 font-semibold">
                  {idx === 0 ? 'Stem Conv Block' : `Deep Conv Block ${idx + 1}`}
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-4">
                {/* Channel Selector */}
                <div className="flex items-center gap-1.5">
                  <span className="text-zinc-500 text-[10px]">CHANNELS:</span>
                  <select
                    value={l.filters}
                    onChange={(e) => handleFilterChange(idx, e.target.value)}
                    className="bg-black/60 border border-white/10 rounded-lg px-2.5 py-1 text-white text-xs font-mono focus:border-cyan-400 focus:outline-none"
                  >
                    <option value="16">16 CH</option>
                    <option value="32">32 CH</option>
                    <option value="64">64 CH</option>
                    <option value="128">128 CH</option>
                    <option value="256">256 CH</option>
                  </select>
                </div>

                {/* Kernel Selector */}
                <div className="flex items-center gap-1.5">
                  <span className="text-zinc-500 text-[10px]">KERNEL:</span>
                  <select
                    value={l.kernel}
                    onChange={(e) => handleKernelChange(idx, e.target.value)}
                    className="bg-black/60 border border-white/10 rounded-lg px-2.5 py-1 text-white text-xs font-mono focus:border-cyan-400 focus:outline-none"
                  >
                    <option value="3">3×3</option>
                    <option value="5">5×5</option>
                    <option value="7">7×7</option>
                  </select>
                </div>

                {/* Delete button */}
                {layers.length > 1 && (
                  <button
                    onClick={() => handleRemoveLayer(idx)}
                    className="p-1.5 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                    title="Remove Layer"
                  >
                    <Trash2 size={13} />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        {layers.length < 5 && (
          <button
            onClick={handleAddLayer}
            className="w-full py-2.5 rounded-xl border border-dashed border-white/10 hover:border-cyan-400/40 text-zinc-400 hover:text-cyan-300 font-mono text-xs flex items-center justify-center gap-1.5 transition-all bg-white/[0.01] hover:bg-white/[0.03]"
          >
            <Plus size={13} /> ADD CONVOLUTIONAL LAYER
          </button>
        )}
      </div>

      {/* Global Hyperparameter Synthesizer Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6 font-mono text-xs">
        {/* Activation */}
        <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/[0.06] space-y-1.5">
          <span className="text-[10px] text-zinc-500 uppercase tracking-wider block">Activation Function</span>
          <div className="flex gap-1.5">
            {['GELU', 'ReLU', 'SiLU', 'Mish'].map((act) => (
              <button
                key={act}
                onClick={() => {
                  soundEngine.tick(1300, 0.01);
                  setActivation(act);
                }}
                className={`flex-1 py-1 rounded-lg text-[11px] font-semibold transition-all border ${
                  activation === act
                    ? 'bg-emerald-500/20 border-emerald-400 text-emerald-300'
                    : 'bg-black/40 border-white/5 text-zinc-400 hover:text-white'
                }`}
              >
                {act}
              </button>
            ))}
          </div>
        </div>

        {/* Batch Normalization */}
        <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/[0.06] space-y-1.5">
          <span className="text-[10px] text-zinc-500 uppercase tracking-wider block">Batch Normalization</span>
          <button
            onClick={() => {
              soundEngine.tick(1200, 0.01);
              setBatchNorm(!batchNorm);
            }}
            className={`w-full py-1 rounded-lg text-[11px] font-semibold transition-all border ${
              batchNorm
                ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                : 'bg-black/40 border-white/5 text-zinc-500'
            }`}
          >
            {batchNorm ? 'ENABLED (STABILIZED)' : 'DISABLED (RAW)'}
          </button>
        </div>

        {/* Dropout */}
        <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/[0.06] space-y-1.5">
          <div className="flex justify-between text-[10px] text-zinc-500 uppercase tracking-wider">
            <span>Dropout Rate</span>
            <span className="text-zinc-300">{Math.round(dropout * 100)}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="0.5"
            step="0.05"
            value={dropout}
            onChange={(e) => setDropout(Number(e.target.value))}
            className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
          />
        </div>
      </div>

      {/* Action Banner & Result Readout */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 rounded-2xl bg-white/[0.03] border border-white/[0.08]">
        <div>
          <div className="font-mono text-xs text-zinc-300 font-semibold flex items-center gap-2">
            <Sparkles size={14} className="text-emerald-400" />
            <span>Ready to challenge AI Champion ({championAcc}% Benchmark)?</span>
          </div>
          <div className="font-mono text-[10px] text-zinc-500 mt-0.5">
            Test whether human intuition can configure a higher-performing topology.
          </div>
        </div>

        <button
          onClick={handleSimulateChallenge}
          disabled={testing}
          className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-400 to-cyan-400 hover:from-emerald-300 hover:to-cyan-300 text-black font-mono text-xs font-bold uppercase tracking-wider shadow-[0_0_20px_rgba(16,185,129,0.3)] transition-all transform active:scale-95 disabled:opacity-50 flex items-center gap-2 shrink-0"
        >
          <Play size={13} fill="currentColor" />
          <span>{testing ? 'COMPUTING ESTIMATE...' : 'CHALLENGE AI ARCHITECT'}</span>
        </button>
      </div>

      {/* Challenge Resolution Feedback */}
      {testResult && (
        <div
          className={`mt-4 p-4 rounded-2xl border font-mono text-xs flex flex-col sm:flex-row items-center justify-between gap-3 ${
            testResult.beatChampion
              ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-200 shadow-[0_0_20px_rgba(16,185,129,0.2)]'
              : 'bg-zinc-900/60 border-white/10 text-zinc-300'
          }`}
        >
          <div>
            <div className="font-bold text-sm flex items-center gap-2">
              {testResult.beatChampion ? '🏆 HUMAN INTUITION PREVAILS!' : '⚡ AI ARCHITECT HOLDS THE CROWN'}
            </div>
            <div className="text-[11px] text-zinc-400 mt-0.5">
              Estimated Topology Accuracy: <strong>{testResult.estimatedAccuracy}%</strong> vs. Champion {championAcc}%. Footprint: {(testResult.params / 1000).toFixed(1)}k parameters.
            </div>
          </div>

          {onApplyConfig && (
            <button
              onClick={() => {
                soundEngine.tick(1600, 0.01);
                onApplyConfig({
                  num_layers: layers.length,
                  filters: layers.map((l) => l.filters),
                  kernel_sizes: layers.map((l) => l.kernel),
                  batch_norm: batchNorm,
                  activation,
                  pooling: true,
                  dropout,
                });
              }}
              className="px-3.5 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-[11px] font-semibold border border-white/15 transition-all"
            >
              Apply to 3D Specimen
            </button>
          )}
        </div>
      )}
    </div>
  );
}
