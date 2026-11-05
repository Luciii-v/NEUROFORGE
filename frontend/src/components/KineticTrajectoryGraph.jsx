import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Award, Layers, Zap } from 'lucide-react';

const fallbackGenerations = [
  { gen: 1, acc: 32.8, f1: 24.7, name: 'Seed generation', desc: 'First population around the default blueprint', params: '244.8k', champion: false },
  { gen: 2, acc: 41.0, f1: 34.1, name: 'Filter mutation', desc: 'A wider second layer enters the family tree', params: '84.6k', champion: false },
  { gen: 3, acc: 41.0, f1: 34.1, name: 'Elite preserved', desc: 'The strongest parent survives unchanged', params: '19.9k', champion: false },
  { gen: 4, acc: 39.3, f1: 31.8, name: 'Dropout shift', desc: 'A more regularized branch is tested', params: '19.9k', champion: false },
  { gen: 5, acc: 42.6, f1: 33.8, name: 'Current best', desc: 'Selected candidate under the proxy budget', params: '19.9k', champion: true },
];

export default function KineticTrajectoryGraph({ 
  generations = fallbackGenerations,
  baselineAccuracy = 31.1,
  onSelectGen,
  currentGen = 5
}) {
  const [hoveredIdx, setHoveredIdx] = useState(null);
  const activeGenerations = generations?.length > 1 ? generations : fallbackGenerations;

  // Normalization for SVG curve
  const width = 840;
  const height = 240;
  const paddingX = 60;
  const paddingY = 40;

  const minAcc = 20;
  const maxAcc = 100;

  const points = activeGenerations.map((g, i) => {
    const x = paddingX + (i / (activeGenerations.length - 1)) * (width - paddingX * 2);
    const y = height - paddingY - ((g.acc - minAcc) / (maxAcc - minAcc)) * (height - paddingY * 2);
    return { x, y, ...g, index: i };
  });

  const baselineY = height - paddingY - ((baselineAccuracy - minAcc) / (maxAcc - minAcc)) * (height - paddingY * 2);

  // Smooth SVG Bezier Path
  const getCurvePath = (pts) => {
    if (pts.length < 2) return '';
    let d = `M ${pts[0].x} ${pts[0].y}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i];
      const p1 = pts[i + 1];
      const cpX1 = p0.x + (p1.x - p0.x) * 0.5;
      const cpY1 = p0.y;
      const cpX2 = p0.x + (p1.x - p0.x) * 0.5;
      const cpY2 = p1.y;
      d += ` C ${cpX1} ${cpY1}, ${cpX2} ${cpY2}, ${p1.x} ${p1.y}`;
    }
    return d;
  };

  const pathD = getCurvePath(points);
  const areaD = `${pathD} L ${points[points.length - 1].x} ${height - paddingY} L ${points[0].x} ${height - paddingY} Z`;

  const activeGenData = activeGenerations.find((g) => g.gen === currentGen) || activeGenerations[activeGenerations.length - 1];
  const accuracyGain = activeGenData.acc - baselineAccuracy;

  return (
    <div className="w-full rounded-3xl p-6 md:p-8 border border-white/[0.08] relative overflow-hidden backdrop-blur-2xl bg-gradient-to-b from-[#0b0e17]/80 to-[#05070c]/90 shadow-2xl">
      {/* Header with Telemetry */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-white/[0.06] mb-6">
        <div>
            <div className="font-mono text-[10px] tracking-[0.2em] text-zinc-500 uppercase mb-1">
            RECORDED SEARCH / GENERATION FITNESS
          </div>
          <div className="font-sans text-xl md:text-2xl font-bold tracking-tight text-white flex items-center gap-3">
            <span>Architecture family tree</span>
            <span className="text-emerald-400 text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/25">
              {accuracyGain >= 0 ? '+' : ''}{accuracyGain.toFixed(1)} pp vs baseline
            </span>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-5 font-mono text-xs text-zinc-400">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#00f0ff]" />
            <span className="text-zinc-200">Genetic NAS Spline</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-4 h-[1px] bg-zinc-600 border-t border-dashed border-zinc-500" />
            <span className="text-zinc-500">Fixed baseline ({baselineAccuracy.toFixed(1)}%)</span>
          </div>
        </div>
      </div>

      {/* SVG Kinetic Spline Graph */}
      <div className="relative w-full aspect-[8/2.4] min-h-[220px]">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
          <defs>
            <linearGradient id="splineGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#a78bfa" />
              <stop offset="50%" stopColor="#00f0ff" />
              <stop offset="100%" stopColor="#00f5a0" />
            </linearGradient>
            <linearGradient id="splineArea" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00f0ff" stopOpacity="0.22" />
              <stop offset="100%" stopColor="#00f0ff" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[30, 50, 70, 90].map((val) => {
            const gy = height - paddingY - ((val - minAcc) / (maxAcc - minAcc)) * (height - paddingY * 2);
            return (
              <g key={val}>
                <line x1={paddingX} y1={gy} x2={width - paddingX} y2={gy} stroke="#181d28" strokeDasharray="3 3" />
                <text x={paddingX - 12} y={gy + 4} fill="#525e75" fontSize="10" fontFamily="monospace" textAnchor="end">
                  {val}%
                </text>
              </g>
            );
          })}

          {/* Baseline Reference Line */}
          <line
            x1={paddingX}
            y1={baselineY}
            x2={width - paddingX}
            y2={baselineY}
            stroke="#475569"
            strokeWidth="1.2"
            strokeDasharray="4 4"
          />

          {/* Area Fill */}
          <path d={areaD} fill="url(#splineArea)" />

          {/* Main Spline Curve */}
          <path
            d={pathD}
            fill="none"
            stroke="url(#splineGradient)"
            strokeWidth="3.5"
            strokeLinecap="round"
          />

          {/* Interactive Scrub Points */}
          {points.map((pt, idx) => {
            const isSelected = pt.gen === currentGen;
            const isHovered = hoveredIdx === idx;
            return (
              <g
                key={idx}
                className="cursor-pointer transition-all"
                onClick={() => onSelectGen && onSelectGen(pt.gen)}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
              >
                {/* Glow ring on hover/select */}
                {(isSelected || isHovered) && (
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r={isSelected ? 16 : 12}
                    fill="none"
                    stroke={pt.champion ? '#00f5a0' : '#00f0ff'}
                    strokeWidth="2"
                    strokeOpacity="0.4"
                    className="animate-ping"
                  />
                )}

                {/* Main Node Dot */}
                <circle
                  cx={pt.x}
                  cy={pt.y}
                  r={isSelected ? 7 : 5}
                  fill={pt.champion ? '#00f5a0' : isSelected ? '#00f0ff' : '#ecf0f8'}
                  stroke="#06080d"
                  strokeWidth="2"
                />

                {/* Value Pill Above Point */}
                <foreignObject
                  x={pt.x - 36}
                  y={pt.y - (isSelected ? 38 : 32)}
                  width="72"
                  height="26"
                  className="pointer-events-none"
                >
                  <div
                    className={`text-center font-mono text-[11px] font-bold rounded-full py-0.5 px-1.5 transition-all ${
                      isSelected
                        ? 'bg-zinc-950 border border-emerald-400 text-emerald-400 shadow-[0_0_12px_rgba(0,245,160,0.4)]'
                        : 'bg-zinc-900/80 border border-white/10 text-zinc-300'
                    }`}
                  >
                    {pt.acc.toFixed(1)}%
                  </div>
                </foreignObject>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Scrubbable Generation Timeline Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mt-4">
        {activeGenerations.map((g) => {
          const isSelected = g.gen === currentGen;
          return (
            <button
              key={g.gen}
              onClick={() => onSelectGen && onSelectGen(g.gen)}
              className={`p-3.5 rounded-2xl text-left transition-all border ${
                isSelected
                  ? 'bg-cyan-500/10 border-cyan-400/50 shadow-[0_0_20px_rgba(0,240,255,0.15)] ring-1 ring-cyan-400/30'
                  : 'bg-white/[0.02] border-white/[0.06] hover:border-white/20 hover:bg-white/[0.05]'
              }`}
            >
              <div className="flex items-center justify-between text-[10px] font-mono mb-1">
                <span className={isSelected ? 'text-cyan-300 font-bold' : 'text-zinc-500'}>
                  GEN 0{g.gen}
                </span>
                {g.champion && (
                  <span className="text-[9px] font-bold px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                    TOP
                  </span>
                )}
              </div>
              <div className="text-lg font-extrabold text-white font-mono tracking-tight">
                {g.acc.toFixed(1)}%
              </div>
              <div className="text-[11px] text-zinc-400 truncate mt-0.5 font-sans font-medium">
                {g.name}
              </div>
              <div className="text-[10px] text-zinc-500 font-mono mt-1">
                Params: {g.params}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
