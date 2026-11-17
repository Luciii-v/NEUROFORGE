import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Swords, 
  Trophy, 
  Zap, 
  Cpu, 
  Sparkles, 
  Flag, 
  ArrowRight, 
  ShieldCheck, 
  Flame, 
  Activity, 
  Gauge, 
  Layers, 
  BarChart2, 
  Filter, 
  CheckCircle2,
  TrendingUp,
  Award
} from 'lucide-react';
import confetti from 'canvas-confetti';
import soundEngine from '../utils/soundEngine';

// Visual identity per contender. Every number comes from /api/comparison.
const STYLES = {
  dynamic_cnn: { name: 'NEUROFORGE Champion', tag: 'MUTATION WINNER', badge: 'CHAMPION', category: 'edge',
    badgeClass: 'bg-emerald-500/20 text-emerald-300 border-emerald-400/40 shadow-[0_0_12px_rgba(16,185,129,0.3)]',
    gradient: 'from-emerald-400 via-cyan-400 to-lime-300', trackBorder: 'border-emerald-500/40' },
  mobilenet_v3: { name: 'MobileNet-V3 Small', tag: 'GOOGLE EDGE STANDARD', badge: 'MOBILE AI', category: 'edge',
    badgeClass: 'bg-sky-500/20 text-sky-300 border-sky-400/30', gradient: 'from-sky-400 to-blue-500', trackBorder: 'border-sky-500/30' },
  shufflenet_v2: { name: 'ShuffleNet-V2 0.5×', tag: 'SUB-1.5M COMPACT', badge: 'FLOPS-OPT', category: 'edge',
    badgeClass: 'bg-amber-500/20 text-amber-300 border-amber-400/30', gradient: 'from-amber-400 to-orange-500', trackBorder: 'border-amber-500/30' },
  efficientnet_b0: { name: 'EfficientNet-B0', tag: 'COMPOUND SCALED', badge: 'EFFICIENT', category: 'heavy',
    badgeClass: 'bg-rose-500/20 text-rose-300 border-rose-400/30', gradient: 'from-rose-400 to-pink-500', trackBorder: 'border-rose-500/30' },
  resnet18: { name: 'ResNet-18 Backbone', tag: 'TRANSFER BASELINE', badge: 'DEEP RESIDUAL', category: 'heavy',
    badgeClass: 'bg-purple-500/20 text-purple-300 border-purple-400/30', gradient: 'from-purple-400 to-indigo-500', trackBorder: 'border-purple-500/30' },
  manual_cnn: { name: 'Manual 2-Layer CNN', tag: 'HUMAN DESIGN', badge: 'FIXED HEURISTIC', category: 'baseline',
    badgeClass: 'bg-zinc-600/30 text-zinc-300 border-zinc-500/30', gradient: 'from-zinc-400 to-zinc-600', trackBorder: 'border-zinc-500/20' },
};

const formatParams = (n) => (!n ? '—' : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : `${Math.round(n / 1e3)}k`);

export default function ModelColiseum({ models = [], datasetName = 'the active dataset' }) {
  const [filter, setFilter] = useState('all'); // 'all' | 'edge' | 'heavy' | 'pareto'
  const [racing, setRacing] = useState(false);
  const [winnerAnnounced, setWinnerAnnounced] = useState(false);

  const baseModels = models
    .filter((row) => row.status === 'ready' && row.metrics && STYLES[row.id])
    .map((row) => {
      const accuracy = Number(row.metrics.accuracy.toFixed(1));
      const latency = row.latency_ms ? Number(row.latency_ms.toFixed(2)) : null;
      return {
        ...STYLES[row.id],
        id: row.id,
        architecture: row.kind,
        accuracy,
        f1: Number((row.metrics.f1 ?? 0).toFixed(1)),
        params: formatParams(row.parameters),
        paramNum: row.parameters || 1,
        latency: latency ?? '—',
        baseThroughput: latency ? Math.round(1000 / latency) : 0,
        targetFinish: accuracy,
        accelExp: 0.8 + (1 - accuracy / 100) * 0.8,
        efficiencyQuotient: ((accuracy / (row.parameters || 1)) * 100000).toFixed(1),
        insight: row.note,
        isChampion: row.id === 'dynamic_cnn',
      };
    });
  const notRun = models.filter((row) => row.status !== 'ready' && STYLES[row.id]).map((row) => STYLES[row.id].name);
  const leader = baseModels.reduce((best, m) => (!best || m.accuracy > best.accuracy ? m : best), null);
  const champion = baseModels.find((m) => m.isChampion);
  const efficiencyLeader = baseModels.reduce((best, m) => (!best || Number(m.efficiencyQuotient) > Number(best.efficiencyQuotient) ? m : best), null);

  // Filtered model list
  const filteredModels = baseModels.filter((m) => {
    if (filter === 'edge') return m.category === 'edge' || m.id === 'dynamic_cnn';
    if (filter === 'heavy') return m.category === 'heavy' || m.id === 'dynamic_cnn';
    return true;
  });

  // Track progress mapping: modelId -> 0..100
  const [raceProgress, setRaceProgress] = useState(
    baseModels.reduce((acc, m) => ({ ...acc, [m.id]: m.targetFinish }), {})
  );

  // Dynamic throughput speeds
  const [liveSpeeds, setLiveSpeeds] = useState(
    baseModels.reduce((acc, m) => ({ ...acc, [m.id]: m.baseThroughput }), {})
  );

  const startBattleRace = () => {
    if (racing) return;
    setRacing(true);
    setWinnerAnnounced(false);

    // Reset progress to 0
    setRaceProgress(baseModels.reduce((acc, m) => ({ ...acc, [m.id]: 0 }), {}));
    soundEngine.inferStart();

    const startTime = Date.now();
    const duration = 2100; // 2.1 second dynamic showdown

    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const t = Math.min(1, elapsed / duration);

      const nextProg = {};
      const nextSpeeds = {};

      baseModels.forEach((m) => {
        const p = Math.min(m.targetFinish, Math.round(Math.pow(t, m.accelExp) * m.targetFinish));
        nextProg[m.id] = p;
        // Jitter live speedometers during the race
        const jitter = Math.round((Math.random() - 0.5) * 40);
        nextSpeeds[m.id] = Math.max(80, Math.round(m.baseThroughput * (0.6 + 0.4 * t) + jitter));
      });

      setRaceProgress(nextProg);
      setLiveSpeeds(nextSpeeds);

      if (t >= 1) {
        clearInterval(interval);
        setRacing(false);
        setWinnerAnnounced(true);
        soundEngine.championChord(true);
        confetti({
          particleCount: 65,
          spread: 70,
          origin: { y: 0.65 },
          colors: ['#00f5a0', '#00f0ff', '#fcd34d', '#c084fc'],
        });
      }
    }, 35);
  };

  return (
    <div className="w-full rounded-3xl p-6 md:p-8 border border-white/[0.08] relative overflow-hidden backdrop-blur-2xl bg-gradient-to-b from-[#0e1424]/90 via-[#080c16]/95 to-[#04060a] shadow-2xl">
      {/* Coliseum Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-5 pb-6 border-b border-white/[0.06] mb-6">
        <div>
          <div className="flex items-center gap-2 font-mono text-[10px] tracking-[0.2em] text-cyan-400 uppercase mb-1">
            <Swords size={13} />
            <span>MODEL COLISEUM // MULTI-MODEL ARENA BENCHMARK</span>
          </div>
          <h3 className="font-sans text-xl md:text-2xl font-black tracking-tight text-white flex items-center gap-2">
            The Multi-Model Showdown
          </h3>
          <p className="font-mono text-xs text-zinc-400 mt-1">
            NEUROFORGE champion against {Math.max(0, baseModels.length - 1)} trained baselines on {datasetName}. Test split, same protocol for every model.
          </p>
        </div>

        {/* Action Buttons & Filter Controls */}
        <div className="flex flex-wrap items-center gap-2.5 shrink-0">
          {/* Category Filter Pills */}
          <div className="flex items-center p-1 rounded-xl bg-black/60 border border-white/[0.08] text-[11px] font-mono">
            <button
              onClick={() => {
                soundEngine.tick(1200, 0.01);
                setFilter('all');
              }}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                filter === 'all'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400/30 font-semibold'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              ALL {baseModels.length}
            </button>
            <button
              onClick={() => {
                soundEngine.tick(1200, 0.01);
                setFilter('edge');
              }}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                filter === 'edge'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-400/30 font-semibold'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              EDGE (&lt;3M)
            </button>
            <button
              onClick={() => {
                soundEngine.tick(1200, 0.01);
                setFilter('heavy');
              }}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                filter === 'heavy'
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-400/30 font-semibold'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              FOUNDATION &amp; HEAVY
            </button>
            <button
              onClick={() => {
                soundEngine.tick(1400, 0.01);
                setFilter(filter === 'pareto' ? 'all' : 'pareto');
              }}
              className={`px-3 py-1.5 rounded-lg flex items-center gap-1 transition-all ${
                filter === 'pareto'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-400/40 font-bold'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <TrendingUp size={12} />
              <span>PARETO FRONTIER</span>
            </button>
          </div>

          <button
            onClick={startBattleRace}
            disabled={racing}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 via-cyan-400 to-teal-400 hover:from-emerald-400 hover:to-cyan-300 text-black font-mono text-xs font-bold uppercase tracking-wider shadow-[0_0_25px_rgba(0,240,255,0.35)] transition-all transform active:scale-95 disabled:opacity-50 flex items-center gap-2"
          >
            <Flag size={14} className={racing ? 'animate-bounce' : ''} />
            <span>{racing ? 'RACING CONTENDERS...' : 'SIMULATE BATTLE RACE'}</span>
          </button>
        </div>
      </div>

      {/* PARETO FRONTIER SPECIAL VIEW */}
      {filter === 'pareto' ? (
        <div className="mb-8 p-6 rounded-2xl bg-gradient-to-b from-amber-950/20 to-black/60 border border-amber-500/30 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-amber-500/20 pb-3">
            <div className="flex items-center gap-2 text-amber-300 font-mono text-xs font-bold uppercase tracking-wider">
              <TrendingUp size={16} />
              <span>Pareto Efficiency Analysis — Accuracy / Parameter Footprint Quotient</span>
            </div>
            <span className="font-mono text-[10px] text-zinc-400">
              Formula: (Accuracy % ÷ Parameter Count) × 100,000 — Higher is Better
            </span>
          </div>

          <div className="space-y-3 pt-2">
            {baseModels
              .slice()
              .sort((a, b) => Number(b.efficiencyQuotient) - Number(a.efficiencyQuotient))
              .map((m, rank) => {
                const quotientNum = Number(m.efficiencyQuotient);
                const maxQuotient = 35;
                const barWidth = Math.min(100, Math.round((quotientNum / maxQuotient) * 100));

                return (
                  <div key={m.id} className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-1.5">
                    <div className="flex items-center justify-between font-mono text-xs">
                      <div className="flex items-center gap-2">
                        <span className={`w-5 h-5 rounded-full flex items-center justify-center font-bold text-[10px] ${
                          rank === 0 ? 'bg-amber-400 text-black' : 'bg-zinc-800 text-zinc-400'
                        }`}>
                          #{rank + 1}
                        </span>
                        <span className="font-bold text-white">{m.name}</span>
                        <span className={`text-[9px] px-2 py-0.5 rounded-full border ${m.badgeClass}`}>
                          {m.badge}
                        </span>
                      </div>

                      <div className="flex items-center gap-4 text-xs">
                        <span className="text-zinc-400 font-mono">{m.params} params</span>
                        <span className="text-zinc-300 font-mono font-semibold">{m.accuracy}% Acc</span>
                        <span className={`font-bold font-mono px-2 py-0.5 rounded ${
                          m.isChampion ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-400/40 text-sm' : 'text-amber-400'
                        }`}>
                          {m.efficiencyQuotient} Score
                        </span>
                      </div>
                    </div>

                    <div className="w-full h-2 rounded-full bg-black/60 overflow-hidden p-0.5 border border-white/5">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          m.isChampion ? 'bg-gradient-to-r from-cyan-400 via-emerald-400 to-lime-300 shadow-[0_0_12px_#00f5a0]' : 'bg-gradient-to-r from-amber-500 to-orange-400'
                        }`}
                        style={{ width: `${barWidth}%` }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>

          <p className="text-[11px] font-mono text-zinc-400 pt-2 border-t border-amber-500/10">
            ★ Most accuracy per parameter: <strong className="text-emerald-300">{efficiencyLeader?.name || '—'}</strong>. Highest raw accuracy: <strong>{leader?.name || '—'}</strong> at {leader?.accuracy ?? '—'}%. Parameter efficiency only matters once accuracy is good enough to use.
          </p>
        </div>
      ) : (
        /* 6-LANE RACETRACK VISUALIZER */
        <div className="space-y-4 mb-8 p-5 rounded-2xl bg-black/50 border border-white/[0.07]">
          <div className="font-mono text-[10px] uppercase tracking-wider text-zinc-500 flex justify-between pb-1 border-b border-white/[0.04]">
            <span className="flex items-center gap-1.5">
              <Activity size={12} className="text-cyan-400" />
              <span>Each lane finishes at its measured test accuracy ({filteredModels.length} contenders)</span>
            </span>
            <span className="flex items-center gap-2">
              <Gauge size={12} className="text-emerald-400" />
              <span>Measured on this machine</span>
            </span>
          </div>

          <div className="space-y-3 pt-1">
            {filteredModels.map((m) => {
              const prog = raceProgress[m.id] ?? m.targetFinish;
              const spd = liveSpeeds[m.id] ?? m.baseThroughput;

              return (
                <div key={m.id} className="space-y-1.5">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between font-mono text-xs gap-1">
                    <div className="flex items-center gap-2 text-white">
                      {m.isChampion ? (
                        <Trophy size={14} className="text-emerald-400 animate-pulse" />
                      ) : (
                        <Cpu size={14} className="text-zinc-500" />
                      )}
                      <span className={m.isChampion ? 'font-bold text-emerald-300' : 'text-zinc-200'}>
                        {m.name}
                      </span>
                      <span className={`text-[9px] px-2 py-0.5 rounded-full border ${m.badgeClass}`}>
                        {m.accuracy}% ACC
                      </span>
                      <span className="text-[10px] text-zinc-500 hidden md:inline">
                        ({m.params})
                      </span>
                    </div>

                    <div className="flex items-center gap-3 text-[11px]">
                      <span className="text-zinc-400 font-mono">
                        {spd} <span className="text-[9px] text-zinc-500">samples/sec</span>
                      </span>
                      <span className={m.isChampion ? 'text-emerald-400 font-bold' : 'text-zinc-400'}>
                        {m.latency}ms / sample
                      </span>
                    </div>
                  </div>

                  <div className={`w-full h-3 rounded-full bg-white/5 overflow-hidden p-0.5 border ${m.trackBorder}`}>
                    <div
                      className={`h-full rounded-full bg-gradient-to-r ${m.gradient} transition-all duration-75 ${
                        m.isChampion ? 'shadow-[0_0_14px_#00f5a0]' : 'opacity-85'
                      }`}
                      style={{ width: `${prog}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Winner Telemetry Banner */}
          {winnerAnnounced && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-400/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3 font-mono text-xs"
            >
              <div className="flex items-center gap-2 text-emerald-300 font-bold">
                <CheckCircle2 size={16} className="text-emerald-400 shrink-0" />
                <span>Winner: {leader?.name} at {leader?.accuracy}% test accuracy{champion && leader && leader.id !== champion.id ? ` (NEUROFORGE champion: ${champion.accuracy}%)` : ''}</span>
              </div>
              <div className="text-[11px] text-zinc-400">
                Throughput: <strong className="text-cyan-300">{leader?.baseThroughput || '—'} samples/sec</strong> on this machine
              </div>
            </motion.div>
          )}
        </div>
      )}

      {notRun.length > 0 && (
        <p className="font-mono text-[11px] text-zinc-500 mb-4">Not trained on this dataset yet: {notRun.join(', ')}. Train them in the mutation loop's compare panel to add their lanes.</p>
      )}
      {!baseModels.length && (
        <p className="font-mono text-xs text-zinc-400 mb-4">No trained models on this dataset yet. Run a search and train the winner to fill the arena.</p>
      )}

      {/* Metrics Comparison Cards Deck */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredModels.map((m) => (
          <div
            key={m.id}
            className={`p-5 rounded-2xl relative overflow-hidden transition-all duration-200 ${
              m.isChampion
                ? 'bg-gradient-to-b from-emerald-950/40 via-cyan-950/20 to-transparent border border-emerald-400/40 shadow-[0_0_20px_rgba(0,245,160,0.12)]'
                : 'bg-white/[0.02] border border-white/[0.06] hover:border-white/[0.12]'
            }`}
          >
            <div className="flex items-center justify-between mb-2.5 font-mono text-[10px] uppercase tracking-widest font-bold">
              <span className={m.isChampion ? 'text-emerald-400' : 'text-zinc-400'}>{m.tag}</span>
              <span className={`w-2 h-2 rounded-full ${m.isChampion ? 'bg-emerald-400 shadow-[0_0_8px_#10b981]' : 'bg-zinc-600'}`} />
            </div>

            <h4 className="font-sans text-base md:text-lg font-bold text-white mb-0.5">{m.name}</h4>
            <p className="font-mono text-[11px] text-zinc-400 mb-3">{m.architecture}</p>

            <div className="space-y-2 font-mono text-xs border-t border-white/[0.06] pt-3">
              <div className="flex justify-between text-zinc-300">
                <span>Test Accuracy:</span>
                <strong className={m.isChampion ? 'text-emerald-400 font-bold text-sm' : 'text-zinc-200 font-bold text-sm'}>
                  {m.accuracy}%
                </strong>
              </div>
              <div className="flex justify-between text-zinc-400">
                <span>Macro F1:</span>
                <strong className="text-zinc-200">{m.f1}%</strong>
              </div>
              <div className="flex justify-between text-zinc-400">
                <span>Model Footprint:</span>
                <strong className="text-cyan-300">{m.params}</strong>
              </div>
              <div className="flex justify-between text-zinc-400">
                <span>MPS Latency:</span>
                <strong className="text-emerald-300">{m.latency}ms</strong>
              </div>
              <div className="flex justify-between text-zinc-400">
                <span>Efficiency Quotient:</span>
                <strong className={m.isChampion ? 'text-amber-300 font-bold' : 'text-zinc-400'}>
                  {m.efficiencyQuotient}
                </strong>
              </div>
            </div>

            <div className={`mt-3.5 pt-2.5 border-t text-[10px] font-mono ${
              m.isChampion ? 'border-emerald-500/20 text-emerald-300/80' : 'border-white/[0.06] text-zinc-500'
            }`}>
              {m.insight}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
