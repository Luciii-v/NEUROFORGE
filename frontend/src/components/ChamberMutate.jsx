import React, { useState, useEffect, useRef } from 'react';
import { Dna, Play, Zap, Award, ArrowRight, RotateCcw } from 'lucide-react';

export default function ChamberMutate({ currentDataset, onSetSpeech, onSearchCompleted }) {
  const [generations, setGenerations] = useState(3);
  const [population, setPopulation] = useState(4);
  const [proxyEpochs, setProxyEpochs] = useState(2);
  const [searching, setSearching] = useState(false);
  const [candidates, setCandidates] = useState([]);
  const [bestCandidate, setBestCandidate] = useState(null);
  const [baseline, setBaseline] = useState(null);
  const [currentGen, setCurrentGen] = useState(1);
  const wsRef = useRef(null);

  // Load existing history on mount
  useEffect(() => {
    fetch('/api/history')
      .then((res) => res.json())
      .then((data) => {
        if (data.history && data.history.length > 0) {
          setCandidates(data.history);
          const bestId = data.metadata?.best_candidate_id;
          const best = data.history.find((c) => c.candidate_id === bestId) || data.history[0];
          setBestCandidate(best);
          setBaseline(data.metadata?.manual_baseline || null);
        }
      })
      .catch((err) => console.log('History fetch error:', err));
  }, []);

  const startForgeSearch = () => {
    if (searching) return;
    setSearching(true);
    setCandidates([]);
    setBestCandidate(null);
    if (onSetSpeech) onSetSpeech(`Igniting the Crucible! Searching across ${generations} generations with ${population} candidates per gen...`, 'forging');

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/search`;
    const socket = new WebSocket(wsUrl);
    wsRef.current = socket;

    socket.onopen = () => {
      socket.send(JSON.stringify({
        generations: Number(generations),
        population: Number(population),
        proxy_epochs: Number(proxyEpochs),
        dataset_dir: currentDataset?.dataset_root,
      }));
    };

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'candidate_evaluated') {
          const cand = msg.data;
          setCandidates((prev) => [cand, ...prev]);
          if (cand.generation) setCurrentGen(cand.generation);
          if (onSetSpeech) onSetSpeech(`Synthesized Candidate #${cand.candidate_id}: Hit ${cand.accuracy?.toFixed(1)}% accuracy with ${cand.filters?.join('-')} filter layout.`, 'forging');
        } else if (msg.type === 'search_completed') {
          setSearching(false);
          setBestCandidate(msg.best_result);
          setBaseline(msg.manual_baseline);
          if (onSetSpeech) onSetSpeech(`Search cycle complete! Top model achieved ${msg.best_result?.accuracy?.toFixed(1)}% accuracy. Ready to forge final weights!`, 'victory');
          if (onSearchCompleted) onSearchCompleted(msg);
        } else if (msg.type === 'error') {
          setSearching(false);
          if (onSetSpeech) onSetSpeech(`The Crucible encountered a flaw: ${msg.message}`, 'caution');
        }
      } catch (e) {
        console.error('WS Parse Error:', e);
      }
    };

    socket.onerror = (err) => {
      console.error('WebSocket Error:', err);
      setSearching(false);
      if (onSetSpeech) onSetSpeech('Crucible connection interrupted.', 'caution');
    };

    socket.onclose = () => {
      setSearching(false);
    };
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-purple-500/10 pb-4">
        <h2 className="font-cinzel text-xl md:text-2xl font-bold tracking-wide text-slate-100 flex items-center gap-3">
          <Dna className="w-6 h-6 text-purple-400" />
          <span>CHAMBER II: THE GENETIC CRUCIBLE</span>
        </h2>
        <p className="text-slate-400 text-sm mt-1">
          Automated Neural Architecture Search (NAS) mutating filters, kernels, activations, and pooling layers over successive generations.
        </p>
      </div>

      {/* Control Console */}
      <div className="atelier-panel rounded-xl p-5 border-purple-500/20 grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
        <div>
          <label className="block font-mono text-[11px] text-purple-300 uppercase tracking-wider mb-1">
            Generations
          </label>
          <input
            type="number"
            min="1"
            max="10"
            value={generations}
            disabled={searching}
            onChange={(e) => setGenerations(e.target.value)}
            className="w-full bg-black/60 border border-purple-500/20 rounded-lg px-3 py-2 text-sm font-mono text-slate-200"
          />
        </div>

        <div>
          <label className="block font-mono text-[11px] text-purple-300 uppercase tracking-wider mb-1">
            Population / Gen
          </label>
          <input
            type="number"
            min="2"
            max="16"
            value={population}
            disabled={searching}
            onChange={(e) => setPopulation(e.target.value)}
            className="w-full bg-black/60 border border-purple-500/20 rounded-lg px-3 py-2 text-sm font-mono text-slate-200"
          />
        </div>

        <div>
          <label className="block font-mono text-[11px] text-purple-300 uppercase tracking-wider mb-1">
            Proxy Epochs
          </label>
          <input
            type="number"
            min="1"
            max="5"
            value={proxyEpochs}
            disabled={searching}
            onChange={(e) => setProxyEpochs(e.target.value)}
            className="w-full bg-black/60 border border-purple-500/20 rounded-lg px-3 py-2 text-sm font-mono text-slate-200"
          />
        </div>

        <div className="pt-4 md:pt-0">
          <button
            onClick={startForgeSearch}
            disabled={searching}
            className="btn-alchemist w-full py-2.5 rounded-lg text-xs font-bold flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {searching ? (
              <>
                <RotateCcw className="w-4 h-4 animate-spin text-amber-950" />
                <span>FORGING GEN {currentGen}...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>IGNITE SEARCH</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Real-Time Evolution Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Candidates Feed */}
        <div className="lg:col-span-7 space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400 pb-1">
            <span>MUTATED POPULATION ({candidates.length} EVALUATED)</span>
            {searching && <span className="text-amber-400 animate-pulse">● LIVE PROXIES RUNNING</span>}
          </div>

          <div className="space-y-2.5 max-h-[460px] overflow-y-auto pr-2">
            {candidates.map((cand, idx) => (
              <div
                key={idx}
                className="atelier-panel atelier-panel-hover rounded-xl p-3.5 border-white/5 flex items-center justify-between font-mono text-xs"
              >
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-amber-400 font-bold">#{cand.candidate_id || `CAND-${idx}`}</span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-purple-500/10 text-purple-300 border border-purple-500/20">
                      Gen {cand.generation || 1}
                    </span>
                    <span className="text-slate-400 text-[10px]">
                      {cand.activation || 'ReLU'} · {cand.num_layers} Layers
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Filters: <span className="text-slate-200">[{cand.filters?.join(', ')}]</span> | Kernels: <span className="text-slate-200">[{cand.kernel_sizes?.join(', ')}]</span>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-sm font-bold text-emerald-400">
                    {cand.accuracy ? `${cand.accuracy.toFixed(1)}%` : '—'}
                  </div>
                  <div className="text-[10px] text-slate-400">
                    {(cand.parameters / 1000).toFixed(0)}k params
                  </div>
                </div>
              </div>
            ))}

            {candidates.length === 0 && (
              <div className="text-center py-16 atelier-panel rounded-xl border-dashed border border-white/10 text-slate-500 font-mono text-xs">
                No active search candidates. Click "IGNITE SEARCH" to breed models.
              </div>
            )}
          </div>
        </div>

        {/* Champion Blueprint Card */}
        <div className="lg:col-span-5">
          <div className="atelier-panel rounded-xl p-5 border-purple-500/30 h-full relative overflow-hidden flex flex-col justify-between">
            <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-2xl pointer-events-none" />

            <div>
              <div className="flex items-center justify-between pb-3 border-b border-purple-500/20 mb-4">
                <span className="font-cinzel text-xs text-purple-300 font-bold uppercase tracking-wider flex items-center gap-2">
                  <Award className="w-4 h-4 text-amber-400" />
                  CHAMPION TOPOLOGY
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-500/20 text-purple-200">
                  {bestCandidate?.candidate_id || 'NONE SELECTED'}
                </span>
              </div>

              {bestCandidate ? (
                <div className="space-y-4 font-mono text-xs">
                  <div className="flex justify-between items-center py-1 border-b border-white/5">
                    <span className="text-slate-400">Proxy Accuracy:</span>
                    <span className="text-emerald-400 font-bold text-sm">
                      {bestCandidate.accuracy?.toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-1 border-b border-white/5">
                    <span className="text-slate-400">Baseline Gain:</span>
                    <span className="text-amber-400 font-bold">
                      +{((bestCandidate.accuracy || 0) - (baseline?.accuracy || 0)).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-1 border-b border-white/5">
                    <span className="text-slate-400">Parameter Count:</span>
                    <span className="text-slate-200">{bestCandidate.parameters?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center py-1 border-b border-white/5">
                    <span className="text-slate-400">Activation:</span>
                    <span className="text-purple-300 font-bold">{bestCandidate.activation || 'ReLU'}</span>
                  </div>
                  <div className="flex justify-between items-center py-1 border-b border-white/5">
                    <span className="text-slate-400">Layer Depth:</span>
                    <span className="text-slate-200">{bestCandidate.num_layers} Conv Layers</span>
                  </div>

                  <div className="p-3 rounded-lg bg-black/60 border border-purple-500/20 text-[11px] text-slate-300">
                    <span className="text-purple-300 font-bold block mb-1">Architecture DNA:</span>
                    <div>Filters: [{bestCandidate.filters?.join(', ')}]</div>
                    <div>Kernels: [{bestCandidate.kernel_sizes?.join(', ')}]</div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-slate-500 text-xs font-mono">
                  The genetic crucible has not chosen a champion yet.
                </div>
              )}
            </div>

            <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span>Ready for Deep Training</span>
              <span className="text-amber-400 font-bold">SHIP CHAMBER →</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
