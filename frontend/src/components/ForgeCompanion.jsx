import React from 'react';
import { Flame, Sparkles, Compass, CheckCircle2, AlertCircle } from 'lucide-react';

export default function ForgeCompanion({ state, speech, mood = 'idle' }) {
  // Moods: 'idle', 'curious', 'forging', 'victory', 'caution'
  
  const getBadge = () => {
    switch (mood) {
      case 'forging':
        return { icon: Flame, text: 'MUTATION ACTIVE', color: 'text-amber-400 bg-amber-500/10 border-amber-500/30' };
      case 'victory':
        return { icon: Sparkles, text: 'GOLD STRUCK', color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' };
      case 'curious':
        return { icon: Compass, text: 'INSPECTING INGREDIENTS', color: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30' };
      case 'caution':
        return { icon: AlertCircle, text: 'ADJUSTING FORMULA', color: 'text-rose-400 bg-rose-500/10 border-rose-500/30' };
      default:
        return { icon: CheckCircle2, text: 'ATELIER ONLINE', color: 'text-purple-400 bg-purple-500/10 border-purple-500/30' };
    }
  };

  const badge = getBadge();
  const Icon = badge.icon;

  return (
    <div className="atelier-panel rounded-2xl p-4 md:p-5 relative overflow-hidden flex flex-col md:flex-row items-center gap-4 md:gap-6 border-amber-500/20 shadow-[0_12px_40px_rgba(0,0,0,0.6)]">
      {/* Warm ambient background flare */}
      <div className="absolute -right-8 -bottom-8 w-36 h-36 bg-amber-500/10 rounded-full blur-2xl pointer-events-none" />

      {/* Forge Character SVG Avatar */}
      <div className="relative shrink-0">
        <div className="w-16 h-16 md:w-20 md:h-20 rounded-2xl bg-gradient-to-br from-[#2a1d17] to-[#0f0e13] border border-amber-500/30 flex items-center justify-center shadow-[0_0_20px_rgba(245,158,11,0.2)]">
          <svg className="w-10 h-10 md:w-12 md:h-12" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            {/* Alchemical Hood / Helm */}
            <path d="M12 28C12 18 17 8 24 8C31 8 36 18 36 28C36 34 31 38 24 38C17 38 12 34 12 28Z" fill="#1b1716" stroke="#b45309" strokeWidth="2" />
            
            {/* Brass Goggles Frame */}
            <circle cx="19" cy="22" r="5" fill="#0f0e13" stroke="#f59e0b" strokeWidth="2" />
            <circle cx="29" cy="22" r="5" fill="#0f0e13" stroke="#f59e0b" strokeWidth="2" />
            <path d="M24 21V23" stroke="#f59e0b" strokeWidth="2" />

            {/* Glowing Lenses reacting to state */}
            {mood === 'forging' ? (
              <>
                <circle cx="19" cy="22" r="3" fill="#f59e0b" className="animate-pulse" />
                <circle cx="29" cy="22" r="3" fill="#f59e0b" className="animate-pulse" />
              </>
            ) : mood === 'victory' ? (
              <>
                <circle cx="19" cy="22" r="3" fill="#10b981" />
                <circle cx="29" cy="22" r="3" fill="#10b981" />
              </>
            ) : (
              <>
                <circle cx="19" cy="22" r="2.5" fill="#c084fc" />
                <circle cx="29" cy="22" r="2.5" fill="#c084fc" />
              </>
            )}

            {/* Brass Rivet Mouth or Spark */}
            <path d="M21 31C22.5 32 25.5 32 27 31" stroke="#d97706" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>

        {/* Small active badge dot */}
        <span className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-amber-500 border-2 border-[#1b1716] shadow-[0_0_10px_#f59e0b] animate-ping" />
        <span className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-amber-500 border-2 border-[#1b1716]" />
      </div>

      {/* Speech & Thought Stream */}
      <div className="flex-1 text-center md:text-left">
        <div className="flex items-center justify-center md:justify-start gap-2 mb-1.5">
          <span className="font-cinzel text-xs font-bold tracking-widest text-amber-300 uppercase">FORGE // STUDIO ALCHEMIST</span>
          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono border ${badge.color}`}>
            <Icon className="w-3 h-3" />
            {badge.text}
          </span>
        </div>

        <p className="text-slate-200 text-sm md:text-base font-medium leading-relaxed font-sans">
          "{speech || 'The furnace is heated and ready. Feed in a dataset to begin the transmutation.'}"
        </p>
      </div>
    </div>
  );
}
