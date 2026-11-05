import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, Zap, CheckCircle2, Cpu, ArrowUpRight, Sparkles, Image as ImageIcon } from 'lucide-react';
import confetti from 'canvas-confetti';

export default function InteractiveCrucible({ onSetSpeech, onInferStart, onInferEnd }) {
  const [testImage, setTestImage] = useState(null);
  const [evaluating, setEvaluating] = useState(false);
  const [result, setResult] = useState(null);
  const [samples, setSamples] = useState([]);
  const [activeSampleName, setActiveSampleName] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Fetch quick-test sample images from backend
    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => {
        if (data.samples && data.samples.length > 0) {
          setSamples(data.samples);
        }
      })
      .catch((err) => console.log('Samples API not available yet', err));
  }, []);

  const handlePredict = async (fileBlob, filename = 'test.jpg') => {
    if (!fileBlob) return;

    setEvaluating(true);
    setResult(null);
    if (onInferStart) onInferStart();

    const formData = new FormData();
    formData.append('file', fileBlob, filename);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Inference failed');

      setResult(data);

      if (data.percentage > 40) {
        confetti({
          particleCount: 45,
          spread: 55,
          origin: { y: 0.8 },
          colors: ['#00f5a0', '#00f0ff', '#a78bfa'],
        });
      }

       if (onSetSpeech) {
         const qualifier = data.percentage < 40 ? ' Low-confidence result; the model is uncertain.' : '';
         onSetSpeech(`Inference returned '${data.prediction}' with ${data.percentage}% confidence in ${data.latency_ms}ms.${qualifier}`);
       }
    } catch (e) {
      console.error(e);
      if (onSetSpeech) onSetSpeech(`Inference error: ${e.message}`);
    } finally {
      setEvaluating(false);
      if (onInferEnd) onInferEnd();
    }
  };

  const handleFileUpload = (file) => {
    if (!file) return;
    setActiveSampleName(file.name);
    setTestImage(URL.createObjectURL(file));
    handlePredict(file, file.name);
  };

  const handleSelectSample = async (sample) => {
    setActiveSampleName(sample.class_name);
    try {
      setEvaluating(true);
      const url = `/api/sample-image?rel_path=${encodeURIComponent(sample.rel_path)}`;
      setTestImage(url);
      
      const res = await fetch(url);
      const blob = await res.blob();
      await handlePredict(blob, sample.filename);
    } catch (err) {
      console.error('Failed to load sample image', err);
      setEvaluating(false);
    }
  };

  return (
    <div className="w-full rounded-3xl p-6 md:p-8 border border-white/[0.08] relative overflow-hidden backdrop-blur-2xl bg-gradient-to-b from-[#0b0e17]/80 to-[#05070c]/90 shadow-2xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 border-b border-white/[0.06] mb-6">
        <div>
          <span className="font-mono text-[10px] tracking-[0.2em] text-zinc-500 uppercase block mb-1">
            CRUCIBLE // REAL-TIME INFERENCE SUITE
          </span>
          <span className="font-sans text-xl md:text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            Interactive Prediction Chamber
          </span>
        </div>
        <div className="flex items-center gap-2 font-mono text-xs text-zinc-400 bg-white/[0.03] px-3 py-1.5 rounded-full border border-white/[0.06] self-start sm:self-auto">
          <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#10b981] animate-pulse" />
          <span className="text-zinc-200">MPS ENGINE ONLINE</span>
        </div>
      </div>

      {/* Quick Test Presets Carousel */}
      {samples.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center justify-between text-xs font-mono text-zinc-400 mb-2.5">
            <span className="text-[10px] tracking-wider uppercase text-zinc-500">Quick Test Specimen Presets:</span>
            <span className="text-[10px] text-zinc-600">Click to evaluate instantly</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {samples.map((s, idx) => {
              const isSelected = activeSampleName === s.class_name;
              return (
                <button
                  key={idx}
                  onClick={() => handleSelectSample(s)}
                  disabled={evaluating}
                  className={`px-3.5 py-1.5 rounded-xl font-mono text-xs transition-all flex items-center gap-2 border ${
                    isSelected
                      ? 'bg-cyan-500/15 border-cyan-400 text-cyan-200 shadow-[0_0_15px_rgba(0,240,255,0.15)]'
                      : 'bg-white/[0.03] border-white/[0.08] text-zinc-300 hover:border-white/20 hover:bg-white/[0.06]'
                  } disabled:opacity-50`}
                >
                  <Sparkles className="w-3 h-3 text-cyan-400" />
                  <span>{s.class_name}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Main Crucible Grid */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
        {/* Dropzone & Preview Box */}
        <div className="md:col-span-5">
          <div
            onClick={() => fileInputRef.current?.click()}
            className="group relative w-full aspect-square rounded-2xl border-2 border-dashed border-white/10 hover:border-cyan-400/40 transition-all flex flex-col items-center justify-center p-6 text-center cursor-pointer bg-black/40 overflow-hidden hover:shadow-[0_0_35px_rgba(0,240,255,0.08)]"
          >
            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFileUpload(e.target.files?.[0])}
            />

            {testImage ? (
              <div className="relative w-full h-full flex flex-col items-center justify-center">
                <img
                  src={testImage}
                  alt="Specimen Preview"
                  className="w-full h-full object-cover rounded-xl shadow-lg border border-white/10"
                />
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center rounded-xl font-mono text-xs text-cyan-300 backdrop-blur-xs gap-1">
                  <ImageIcon className="w-5 h-5 text-cyan-400" />
                  <span>DROP OR CLICK TO SWAP</span>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center">
                <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-400/20 flex items-center justify-center text-cyan-400 mb-4 group-hover:scale-110 group-hover:shadow-[0_0_20px_rgba(0,240,255,0.2)] transition-all">
                  <UploadCloud className="w-7 h-7" />
                </div>
                <div className="font-sans text-sm font-semibold text-zinc-100">Drop Custom Test Image</div>
                <div className="font-mono text-[11px] text-zinc-500 mt-1">PNG, JPG, or WEBP (Direct Inference)</div>
              </div>
            )}

            {evaluating && (
              <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
                <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                <span className="font-mono text-xs text-cyan-300 tracking-wider">COMPUTING FORWARD PASS...</span>
              </div>
            )}
          </div>
        </div>

        {/* Live Output HUD */}
        <div className="md:col-span-7 space-y-4">
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                className="space-y-4"
              >
                {/* Top Prediction Banner */}
                <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-md">
                  <div className="flex items-center justify-between text-xs font-mono text-zinc-500 mb-1">
                    <span>IDENTIFIED CLASS</span>
                    <span className="text-cyan-400 flex items-center gap-1">
                      <Zap className="w-3 h-3 fill-current" />
                      {result.latency_ms}ms LATENCY
                    </span>
                  </div>
                  <div className="text-2xl md:text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
                    <span>{result.prediction}</span>
                    <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                      {result.percentage}% CONFIDENCE
                    </span>
                  </div>
                </div>

                {result.percentage < 40 && (
                  <div className="rounded-2xl border border-amber-400/30 bg-amber-400/[0.06] px-5 py-3 text-xs font-mono text-amber-200">
                    <span className="font-semibold">Low-confidence reading.</span>{' '}
                    Five classes give a 20% random baseline. This {result.percentage}% result is close to that baseline, so treat the label as uncertain—not as a reliable identification.
                  </div>
                )}

                {/* Probability Spectrum Breakdown */}
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/[0.06] space-y-3">
                  <div className="font-mono text-[10px] tracking-wider uppercase text-zinc-500 flex justify-between">
                    <span>Probability Distribution Across 5 Classes</span>
                    <span>Softmax Probability</span>
                  </div>

                  <div className="space-y-2.5">
                    {result.all_classes?.map((item, idx) => {
                      const isTop = idx === 0;
                      return (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between font-mono text-xs">
                            <span className={isTop ? 'text-zinc-100 font-semibold' : 'text-zinc-400'}>
                              {item.class_name}
                            </span>
                            <span className={isTop ? 'text-emerald-400 font-bold' : 'text-zinc-500'}>
                              {item.percentage}%
                            </span>
                          </div>
                          <div className="w-full h-1.5 rounded-full bg-white/5 overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${item.percentage}%` }}
                              transition={{ duration: 0.5, ease: 'easeOut' }}
                              className={`h-full rounded-full ${
                                isTop ? 'bg-gradient-to-r from-cyan-400 to-emerald-400' : 'bg-zinc-600'
                              }`}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </motion.div>
            ) : (
              <div className="h-[280px] rounded-2xl bg-white/[0.02] border border-white/[0.06] flex flex-col items-center justify-center p-8 text-center text-zinc-500 font-mono text-xs space-y-2">
                <Cpu className="w-8 h-8 text-zinc-600" />
                <div className="text-zinc-300 font-sans font-semibold text-sm">Awaiting Specimen Input</div>
                <div className="max-w-sm text-zinc-500">
                  Select one of the sample presets above or drop an image to execute real-time tensor inference directly through the Apple M5 accelerator.
                </div>
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
