import React, { useState, useRef } from 'react';
import { Award, Zap, Cpu, Sparkles, Download, Copy, Check, Share2, Layers, ShieldCheck } from 'lucide-react';
import soundEngine from '../utils/soundEngine';

export default function HolographicSpecimenCard({
  modelName = 'NEURO-TITAN // G5-CHAMPION',
  accuracy = '—',
  f1 = '—',
  latency = 1.83,
  paramCount = '—',
  datasetName = 'Fruits Classification (9 Classes)',
  numClasses = 9,
  generation = 5,
  config = null,
  onClose = null,
}) {
  const cardRef = useRef(null);
  const [rotateX, setRotateX] = useState(0);
  const [rotateY, setRotateY] = useState(0);
  const [glarePos, setGlarePos] = useState({ x: 50, y: 50 });
  const [copied, setCopied] = useState(false);

  const handlePointerMove = (e) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const rX = ((y - centerY) / centerY) * -14;
    const rY = ((x - centerX) / centerX) * 14;

    setRotateX(rX);
    setRotateY(rY);
    setGlarePos({
      x: (x / rect.width) * 100,
      y: (y / rect.height) * 100,
    });
  };

  const handlePointerLeave = () => {
    setRotateX(0);
    setRotateY(0);
    setGlarePos({ x: 50, y: 50 });
  };

  const shareText = `🏆 NEUROFORGE Evolved Model Champion: ${modelName}\n• Accuracy: ${accuracy}%\n• Macro F1: ${f1}%\n• Inference Latency: ${latency}ms (Apple Silicon MPS)\n• Parameters: ${paramCount}\nForged autonomously on hardware. #NEUROFORGE #DeepLearning`;

  const handleCopyShare = () => {
    soundEngine.tick(1600, 0.015);
    navigator.clipboard.writeText(shareText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2200);
  };

  return (
    <div className="flex flex-col items-center justify-center p-4">
      {/* 3D Tilted Card Container */}
      <div
        className="perspective-[1000px] w-full max-w-[340px] select-none"
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
      >
        <div
          ref={cardRef}
          style={{
            transform: `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`,
            transition: 'transform 0.15s ease-out',
          }}
          className="relative w-full rounded-3xl p-6 overflow-hidden border border-emerald-400/30 bg-gradient-to-br from-[#0a121e] via-[#060a12] to-[#04060a] shadow-[0_25px_50px_rgba(0,0,0,0.9),0_0_40px_rgba(16,185,129,0.18)]"
        >
          {/* Holographic Rainbow Sheen Foil Layer */}
          <div
            className="absolute inset-0 pointer-events-none transition-opacity duration-300 mix-blend-color-dodge opacity-45"
            style={{
              background: `radial-gradient(circle at ${glarePos.x}% ${glarePos.y}%, rgba(0, 240, 255, 0.6) 0%, rgba(167, 139, 250, 0.45) 30%, rgba(217, 249, 157, 0.35) 60%, transparent 80%)`,
            }}
          />

          {/* Prismatic Diagonal Stripe Overlay */}
          <div className="absolute inset-0 pointer-events-none opacity-20 bg-[repeating-linear-gradient(45deg,transparent,transparent_10px,rgba(255,255,255,0.08)_10px,rgba(255,255,255,0.08)_20px)]" />

          {/* Top Brand & Rarity Tag */}
          <div className="relative z-10 flex items-center justify-between border-b border-white/[0.08] pb-3 mb-4">
            <div className="flex items-center gap-1.5 font-mono text-[10px] text-emerald-400 tracking-widest uppercase font-bold">
              <ShieldCheck size={13} className="text-emerald-400" />
              <span>NEUROFORGE // SSR SPECIMEN</span>
            </div>
            <span className="font-mono text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-400/40 px-2 py-0.5 rounded-full font-bold shadow-[0_0_10px_rgba(16,185,129,0.3)]">
              RANK S // CHAMPION
            </span>
          </div>

          {/* Specimen Title & Crest */}
          <div className="relative z-10 mb-5">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-emerald-500/20 via-cyan-500/20 to-purple-500/20 border border-emerald-400/40 flex items-center justify-center text-emerald-300 mb-3 shadow-[0_0_20px_rgba(16,185,129,0.3)]">
              <Award size={24} />
            </div>
            <h4 className="font-sans text-lg font-black text-white tracking-tight leading-tight">
              {modelName}
            </h4>
            <div className="font-mono text-[10px] text-zinc-400 mt-0.5">
              AUTONOMOUS GENETIC EVOLUTION · GEN {generation}
            </div>
          </div>

          {/* Core Stats Grid */}
          <div className="relative z-10 grid grid-cols-2 gap-2.5 mb-5 font-mono">
            <div className="p-3 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
              <span className="text-[9px] text-zinc-500 uppercase tracking-wider block">TEST ACCURACY</span>
              <strong className="text-xl text-emerald-400 font-extrabold">{accuracy}%</strong>
            </div>

            <div className="p-3 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
              <span className="text-[9px] text-zinc-500 uppercase tracking-wider block">INFERENCE LATENCY</span>
              <strong className="text-xl text-cyan-400 font-extrabold">{latency}ms</strong>
            </div>

            <div className="p-3 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
              <span className="text-[9px] text-zinc-500 uppercase tracking-wider block">MACRO F1 SCORE</span>
              <strong className="text-lg text-zinc-200 font-bold">{f1}%</strong>
            </div>

            <div className="p-3 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
              <span className="text-[9px] text-zinc-500 uppercase tracking-wider block">MODEL PARAMS</span>
              <strong className="text-lg text-zinc-200 font-bold">{paramCount}</strong>
            </div>
          </div>

          {/* Specimen DNA Hash & Architecture Blueprint */}
          <div className="relative z-10 p-3 rounded-2xl bg-black/40 border border-white/[0.06] mb-5 font-mono text-[10px] space-y-1 text-zinc-400">
            <div className="flex justify-between">
              <span className="text-zinc-500 uppercase">TOPOLOGY:</span>
              <span className="text-zinc-300 font-semibold">{config?.num_layers || 3} Layers · GELU · BatchNorm</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500 uppercase">CHANNELS:</span>
              <span className="text-cyan-400 font-semibold">{JSON.stringify(config?.filters || [32, 64, 128])}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500 uppercase">DATASET:</span>
              <span className="text-zinc-300 truncate max-w-[140px]">{datasetName}</span>
            </div>
          </div>

          {/* Footer Receipt */}
          <div className="relative z-10 flex items-center justify-between text-[9px] font-mono text-zinc-500 border-t border-white/[0.06] pt-3">
            <span>SEALED TEST SPLIT</span>
            <span className="text-emerald-400/90 font-semibold">ZERO RESNET WEIGHTS</span>
          </div>
        </div>
      </div>

      {/* Share / Copy Action Bar */}
      <div className="mt-4 flex items-center gap-2">
        <button
          onClick={handleCopyShare}
          className="px-4 py-2 rounded-xl bg-white/[0.06] hover:bg-white/[0.12] text-zinc-200 font-mono text-xs flex items-center gap-1.5 border border-white/10 transition-all shadow-md active:scale-95"
        >
          {copied ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
          <span>{copied ? 'Card Stats Copied!' : 'Copy Specimen Card Text'}</span>
        </button>
      </div>
    </div>
  );
}
