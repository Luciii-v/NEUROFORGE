import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  UploadCloud, 
  Zap, 
  Cpu, 
  Sparkles, 
  Image as ImageIcon, 
  Volume2, 
  Music, 
  Headphones, 
  Eye, 
  EyeOff, 
  Camera, 
  CameraOff, 
  ShieldAlert, 
  Sliders,
  RotateCcw,
  AlertTriangle
} from 'lucide-react';
import confetti from 'canvas-confetti';
import soundEngine from '../utils/soundEngine';

export default function InteractiveCrucible({ onSetSpeech, onInferStart, onInferEnd, onInferResult }) {
  const [testMedia, setTestMedia] = useState(null); // { url, type: 'image' | 'audio', name, blob }
  const [evaluating, setEvaluating] = useState(false);
  const [result, setResult] = useState(null);
  const [inferError, setInferError] = useState(null);
  const [samples, setSamples] = useState([]);
  const [activeSampleName, setActiveSampleName] = useState(null);
  
  // Hook 1: Grad-CAM Mind Reader Saliency state
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.72);

  // Hook 2: Adversarial Stress Test state
  const [noiseLevel, setNoiseLevel] = useState(0);
  const rawImageBlobRef = useRef(null);

  // Hook 2b: Live Webcam Scanner state
  const [cameraActive, setCameraActive] = useState(false);
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);

  const fileInputRef = useRef(null);
  const noiseDebounceTimer = useRef(null);

  const setVideoRef = useCallback((node) => {
    videoRef.current = node;
    if (node && mediaStreamRef.current) {
      node.srcObject = mediaStreamRef.current;
      node.play().catch((err) => console.warn('Video play error on ref attach:', err));
    }
  }, []);

  useEffect(() => {
    if (cameraActive && videoRef.current && mediaStreamRef.current) {
      videoRef.current.srcObject = mediaStreamRef.current;
      videoRef.current.play().catch((err) => console.warn('Video play error on cameraActive change:', err));
    }
  }, [cameraActive]);

  useEffect(() => {
    // Fetch quick-test sample files from backend
    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => {
        if (data.samples && data.samples.length > 0) {
          setSamples(data.samples);
        }
      })
      .catch((err) => console.log('Samples API not available yet', err));

    return () => {
      stopCamera();
    };
  }, []);

  const stopCamera = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  };

  const startCamera = async () => {
    try {
      soundEngine.tick(1200, 0.01);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      });
      mediaStreamRef.current = stream;
      setCameraActive(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch((err) => console.warn('Play error:', err));
      }
      if (onSetSpeech) onSetSpeech('Webcam viewfinder online. Position a target object and click Snap & Infer.');
    } catch (err) {
      alert(`Camera access denied or unavailable: ${err.message}`);
      setCameraActive(false);
    }
  };

  const captureCameraFrame = () => {
    if (!videoRef.current) return;
    const v = videoRef.current;
    const vw = v.videoWidth || 640;
    const vh = v.videoHeight || 480;
    soundEngine.tick(1600, 0.015);
    const canvas = document.createElement('canvas');
    canvas.width = vw;
    canvas.height = vh;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(v, 0, 0, vw, vh);

    canvas.toBlob((blob) => {
      if (!blob) return;
      rawImageBlobRef.current = blob;
      setNoiseLevel(0);
      const url = URL.createObjectURL(blob);
      setTestMedia({
        url,
        type: 'image',
        name: 'webcam-capture.jpg',
        blob,
      });
      stopCamera();
      handlePredict(blob, 'webcam-capture.jpg');
    }, 'image/jpeg', 0.92);
  };

  const handlePredict = async (fileBlob, filename = 'specimen.jpg') => {
    if (!fileBlob) return;

    setEvaluating(true);
    setInferError(null);
    setResult(null);
    soundEngine.inferStart();
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
      if (onInferResult) onInferResult(data);

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
        const modalityTag = data.modality === 'audio' ? ' (Audio Log-Mel Spectrogram)' : '';
        const saliencyTag = data.saliency_map ? ' Grad-CAM attention heatmap computed.' : '';
        onSetSpeech(`Inference returned '${data.prediction}'${modalityTag} with ${data.percentage}% confidence in ${data.latency_ms}ms.${qualifier}${saliencyTag}`);
      }
    } catch (e) {
      console.error(e);
      setInferError(e.message || 'Inference computation failed.');
      if (onSetSpeech) onSetSpeech(`Inference error: ${e.message}`);
    } finally {
      setEvaluating(false);
      if (onInferEnd) onInferEnd();
    }
  };

  const handleFileUpload = (file) => {
    if (!file) return;
    stopCamera();
    soundEngine.tick(1100, 0.015);
    setActiveSampleName(file.name);
    rawImageBlobRef.current = file;
    setNoiseLevel(0);

    const isAudio = file.name.toLowerCase().endsWith('.wav') || file.type.includes('audio');
    setTestMedia({
      url: URL.createObjectURL(file),
      type: isAudio ? 'audio' : 'image',
      name: file.name,
      blob: file,
    });
    handlePredict(file, file.name);
  };

  const handleSelectSample = async (sample) => {
    stopCamera();
    soundEngine.tick(1400, 0.01);
    setActiveSampleName(sample.class_name);
    setNoiseLevel(0);
    try {
      setEvaluating(true);
      const isAudio = sample.modality === 'audio' || sample.filename?.toLowerCase().endsWith('.wav');
      const url = isAudio
        ? `/api/sample-audio?rel_path=${encodeURIComponent(sample.rel_path)}`
        : `/api/sample-image?rel_path=${encodeURIComponent(sample.rel_path)}`;

      const res = await fetch(url);
      const blob = await res.blob();
      rawImageBlobRef.current = blob;

      setTestMedia({
        url,
        type: isAudio ? 'audio' : 'image',
        name: sample.filename,
        blob,
      });

      await handlePredict(blob, sample.filename);
    } catch (err) {
      console.error('Failed to load sample specimen', err);
      setEvaluating(false);
    }
  };

  // Inject Gaussian pixel stress noise into current image
  const handleNoiseChange = (e) => {
    const level = Number(e.target.value);
    setNoiseLevel(level);

    if (!rawImageBlobRef.current || testMedia?.type !== 'image') return;

    if (noiseDebounceTimer.current) clearTimeout(noiseDebounceTimer.current);

    noiseDebounceTimer.current = setTimeout(() => {
      soundEngine.tick(1000 + level * 10, 0.008);
      if (level === 0) {
        handlePredict(rawImageBlobRef.current, testMedia.name || 'clean.jpg');
        return;
      }

      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width || 256;
        canvas.height = img.height || 256;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;
        const intensity = (level / 100) * 160;

        for (let i = 0; i < data.length; i += 4) {
          const noise = (Math.random() - 0.5) * intensity;
          data[i] = Math.min(255, Math.max(0, data[i] + noise));
          data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
          data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
        }

        ctx.putImageData(imgData, 0, 0);
        canvas.toBlob((noisyBlob) => {
          if (noisyBlob) {
            handlePredict(noisyBlob, `stressed_${level}pct.jpg`);
          }
        }, 'image/jpeg', 0.9);
      };
      img.src = URL.createObjectURL(rawImageBlobRef.current);
    }, 180);
  };

  const numClasses = result?.all_classes?.length || 9;
  const randomBaseline = Math.round(100 / numClasses);

  return (
    <div className="w-full rounded-3xl p-6 md:p-8 border border-white/[0.08] relative overflow-hidden backdrop-blur-2xl bg-gradient-to-b from-[#0b0e17]/80 to-[#05070c]/90 shadow-2xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 border-b border-white/[0.06] mb-6">
        <div>
          <span className="font-mono text-[10px] tracking-[0.2em] text-zinc-500 uppercase block mb-1">
            CRUCIBLE // MULTIMODAL INFERENCE SUITE
          </span>
          <span className="font-sans text-xl md:text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            Interactive Prediction Chamber
          </span>
        </div>
        <div className="flex items-center gap-3">
          {/* Webcam Toggle Button */}
          <button
            onClick={() => (cameraActive ? stopCamera() : startCamera())}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full font-mono text-xs border transition-all ${
              cameraActive
                ? 'bg-red-500/15 border-red-400 text-red-300 shadow-[0_0_15px_rgba(239,68,68,0.25)]'
                : 'bg-white/[0.03] hover:bg-white/[0.08] text-zinc-300 border-white/[0.08]'
            }`}
          >
            {cameraActive ? <CameraOff size={13} className="text-red-400" /> : <Camera size={13} className="text-cyan-400" />}
            <span>{cameraActive ? 'CLOSE CAMERA' : 'SCAN LIVE WEBCAM'}</span>
          </button>

          <div className="flex items-center gap-2 font-mono text-xs text-zinc-400 bg-white/[0.03] px-3 py-1.5 rounded-full border border-white/[0.06] self-start sm:self-auto">
            <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#10b981] animate-pulse" />
            <span className="text-zinc-200">MPS ENGINE ONLINE</span>
          </div>
        </div>
      </div>

      {/* Quick Test Presets Carousel */}
      {samples.length > 0 && !cameraActive && (
        <div className="mb-6">
          <div className="flex items-center justify-between text-xs font-mono text-zinc-400 mb-2.5">
            <span className="text-[10px] tracking-wider uppercase text-zinc-500">Quick Test Specimen Presets:</span>
            <span className="text-[10px] text-zinc-600">Click to evaluate instantly</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {samples.map((s, idx) => {
              const isSelected = activeSampleName === s.class_name;
              const isAudio = s.modality === 'audio' || s.filename?.toLowerCase().endsWith('.wav');
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
                  {isAudio ? (
                    <Volume2 className="w-3 h-3 text-cyan-400" />
                  ) : (
                    <Sparkles className="w-3 h-3 text-cyan-400" />
                  )}
                  <span>{s.class_name}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Main Crucible Grid */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
        {/* Dropzone & Live Specimen Viewfinder */}
        <div className="md:col-span-5 space-y-3">
          <div
            onClick={() => !cameraActive && fileInputRef.current?.click()}
            className={`group relative w-full aspect-square rounded-2xl border-2 border-dashed ${
              cameraActive ? 'border-cyan-400/60 shadow-[0_0_30px_rgba(0,240,255,0.15)]' : 'border-white/10 hover:border-cyan-400/40'
            } transition-all flex flex-col items-center justify-center p-4 text-center ${cameraActive ? 'cursor-default' : 'cursor-pointer'} bg-black/40 overflow-hidden`}
          >
            <input
              type="file"
              ref={fileInputRef}
              accept="image/*,audio/wav,audio/x-wav,.wav"
              className="hidden"
              onChange={(e) => handleFileUpload(e.target.files?.[0])}
            />

            {/* LIVE WEBCAM VIEWFINDER HUD */}
            {cameraActive ? (
              <div className="relative w-full h-full rounded-xl overflow-hidden bg-black flex flex-col items-center justify-center">
                <video
                  ref={setVideoRef}
                  autoPlay
                  playsInline
                  muted
                  onLoadedMetadata={() => {
                    if (videoRef.current) {
                      videoRef.current.play().catch((err) => console.warn('Safari video play onLoadedMetadata:', err));
                    }
                  }}
                  className="w-full h-full object-cover rounded-xl"
                />
                {/* Cybernetic Reticle Overlays */}
                <div className="absolute inset-4 pointer-events-none border border-cyan-400/30 rounded-lg flex flex-col justify-between p-2">
                  <div className="flex justify-between font-mono text-[9px] text-cyan-400/70">
                    <span>LIVE SPECIMEN ACQUISITION</span>
                    <span className="animate-pulse">REC ●</span>
                  </div>
                  <div className="w-12 h-12 border-2 border-cyan-400/50 rounded-full self-center flex items-center justify-center">
                    <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-ping" />
                  </div>
                  <div className="flex justify-between font-mono text-[9px] text-cyan-400/70">
                    <span>MPS REAL-TIME HUD</span>
                    <span>60 FPS</span>
                  </div>
                </div>

                {/* Scanline Animation */}
                <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-cyan-400/10 to-transparent h-16 w-full animate-[scan_2.5s_ease-in-out_infinite]" />

                {/* Capture Button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    captureCameraFrame();
                  }}
                  className="absolute bottom-4 z-20 px-5 py-2 rounded-xl bg-cyan-400 hover:bg-cyan-300 text-black font-mono text-xs font-bold tracking-wider uppercase shadow-[0_0_20px_#00f0ff] transition-all transform active:scale-95 flex items-center gap-2"
                >
                  <Camera size={14} />
                  <span>SNAP & INFER TENSOR</span>
                </button>
              </div>
            ) : testMedia ? (
              testMedia.type === 'audio' ? (
                <div
                  className="relative w-full h-full flex flex-col items-center justify-center p-4 bg-zinc-950/80 rounded-xl border border-white/10"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-400/20 flex items-center justify-center text-cyan-400 mb-3 shadow-[0_0_20px_rgba(0,240,255,0.15)]">
                    <Headphones className="w-7 h-7" />
                  </div>
                  <div className="font-mono text-xs text-zinc-200 font-semibold truncate max-w-full mb-1">
                    {testMedia.name || 'audio-specimen.wav'}
                  </div>
                  <span className="font-mono text-[10px] text-zinc-500 uppercase tracking-widest mb-4">
                    16 kHz Mono WAV
                  </span>

                  {/* Animated Waveform Visualizer */}
                  <div className="flex items-center gap-1 h-8 mb-4">
                    {[12, 24, 18, 30, 15, 28, 20, 14, 26, 16, 22].map((h, i) => (
                      <div
                        key={i}
                        className="w-1 bg-cyan-400/70 rounded-full animate-pulse"
                        style={{
                          height: `${h}px`,
                          animationDelay: `${i * 90}ms`,
                        }}
                      />
                    ))}
                  </div>

                  <audio
                    src={testMedia.url}
                    controls
                    className="w-full max-w-[240px] h-8 accent-cyan-400 opacity-90"
                  />

                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="mt-4 font-mono text-[10px] text-cyan-400 hover:text-cyan-300 underline tracking-wider"
                  >
                    CLICK TO SWAP SPECIMEN
                  </button>
                </div>
              ) : (
                <div className="relative w-full h-full flex flex-col items-center justify-center overflow-hidden rounded-xl">
                  {/* Clean Raw Specimen Image */}
                  <img
                    src={testMedia.url}
                    alt="Specimen Preview"
                    className="w-full h-full object-cover rounded-xl shadow-lg border border-white/10"
                  />

                  {/* GRAD-CAM SALIENCY HEATMAP OVERLAY */}
                  {result?.saliency_map && showHeatmap && (
                    <img
                      src={result.saliency_map}
                      alt="Grad-CAM Saliency Heatmap"
                      style={{
                        opacity: heatmapOpacity,
                        mixBlendMode: 'screen',
                      }}
                      className="absolute inset-0 w-full h-full object-cover rounded-xl pointer-events-none transition-opacity duration-200"
                    />
                  )}

                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center rounded-xl font-mono text-xs text-cyan-300 backdrop-blur-xs gap-1 pointer-events-none">
                    <ImageIcon className="w-5 h-5 text-cyan-400" />
                    <span>DROP OR CLICK TO SWAP</span>
                  </div>
                </div>
              )
            ) : (
              <div className="flex flex-col items-center">
                <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-400/20 flex items-center justify-center text-cyan-400 mb-4 group-hover:scale-110 group-hover:shadow-[0_0_20px_rgba(0,240,255,0.2)] transition-all">
                  <UploadCloud className="w-7 h-7" />
                </div>
                <div className="font-sans text-sm font-semibold text-zinc-100">Drop Custom Test Specimen</div>
                <div className="font-mono text-[11px] text-zinc-500 mt-1">Image (PNG, JPG, WEBP) or Audio (.WAV)</div>
                <div className="mt-4 flex items-center gap-2">
                  <span className="font-mono text-[10px] text-zinc-400 bg-white/[0.05] px-2.5 py-1 rounded-md border border-white/10">
                    OR USE SCANNER ABOVE
                  </span>
                </div>
              </div>
            )}

            {evaluating && (
              <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3 z-30">
                <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                <span className="font-mono text-xs text-cyan-300 tracking-wider">COMPUTING FORWARD PASS...</span>
              </div>
            )}
          </div>

          {/* GRAD-CAM SALIENCY CONTROLS */}
          {testMedia?.type === 'image' && result?.saliency_map && (
            <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-md space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      soundEngine.tick(1400, 0.01);
                      setShowHeatmap(!showHeatmap);
                    }}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg font-mono text-[11px] font-semibold border transition-all ${
                      showHeatmap
                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-400/40 shadow-[0_0_12px_rgba(16,185,129,0.2)]'
                        : 'bg-white/[0.04] text-zinc-400 border-white/[0.08]'
                    }`}
                  >
                    {showHeatmap ? <Eye size={12} className="text-emerald-400" /> : <EyeOff size={12} />}
                    <span>MIND READER (GRAD-CAM)</span>
                  </button>
                  <span className="font-mono text-[10px] text-zinc-500">
                    {showHeatmap ? `${Math.round(heatmapOpacity * 100)}% OPAQUE` : 'DISABLED'}
                  </span>
                </div>

                {/* Heat legend */}
                <div className="flex items-center gap-1.5 font-mono text-[9px] text-zinc-500">
                  <span>COLD</span>
                  <div className="w-12 h-2 rounded bg-gradient-to-r from-blue-500 via-yellow-400 to-red-500 opacity-80" />
                  <span>HOT</span>
                </div>
              </div>

              {showHeatmap && (
                <div className="flex items-center gap-3 pt-1">
                  <Sliders size={12} className="text-zinc-500 shrink-0" />
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={heatmapOpacity}
                    onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                    className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
                  />
                </div>
              )}
            </div>
          )}

          {/* ADVERSARIAL STRESS TEST SLIDER */}
          {testMedia?.type === 'image' && (
            <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/[0.06] backdrop-blur-md space-y-2">
              <div className="flex items-center justify-between font-mono text-[11px]">
                <div className="flex items-center gap-1.5 text-zinc-400">
                  <ShieldAlert size={13} className={noiseLevel > 35 ? 'text-amber-400 animate-pulse' : 'text-zinc-500'} />
                  <span className="font-semibold text-zinc-300 uppercase tracking-wider">Adversarial Stress Noise:</span>
                  <span className={noiseLevel > 40 ? 'text-amber-400 font-bold' : 'text-cyan-400'}>{noiseLevel}%</span>
                </div>
                {noiseLevel > 0 && (
                  <button
                    onClick={() => {
                      setNoiseLevel(0);
                      if (rawImageBlobRef.current) handlePredict(rawImageBlobRef.current, 'reset.jpg');
                    }}
                    className="text-[10px] text-zinc-500 hover:text-zinc-300 flex items-center gap-1 underline"
                  >
                    <RotateCcw size={10} /> RESET
                  </button>
                )}
              </div>

              <input
                type="range"
                min="0"
                max="80"
                step="5"
                value={noiseLevel}
                onChange={handleNoiseChange}
                className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />

              <div className="flex justify-between font-mono text-[9px] text-zinc-500">
                <span>0% CLEAN</span>
                <span className={noiseLevel > 40 ? 'text-amber-300 font-semibold' : ''}>
                  {noiseLevel === 0 ? 'NOMINAL ACCURACY' : noiseLevel < 35 ? 'RESILIENT TO NOISE' : 'ADVERSARIAL PERTURBATION'}
                </span>
                <span>80% DEGRADED</span>
              </div>
            </div>
          )}
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
                    <div className="flex items-center gap-2">
                      <span>IDENTIFIED CLASS</span>
                      {result.modality === 'audio' && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-400/20">
                          AUDIO LOG-MEL
                        </span>
                      )}
                      {result.saliency_map && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-400/20 flex items-center gap-1">
                          <Eye size={10} /> GRAD-CAM READY
                        </span>
                      )}
                    </div>
                    <span className="text-cyan-400 flex items-center gap-1 font-mono">
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

                  {result.modality === 'audio' && result.preprocessing && (
                    <div className="mt-3 pt-3 border-t border-white/[0.06] flex items-center justify-between font-mono text-[11px] text-zinc-400">
                      <span>SPECTRUM: <strong className="text-zinc-200">{JSON.stringify(result.feature_shape)}</strong></span>
                      <span className="text-zinc-500">{result.preprocessing}</span>
                    </div>
                  )}
                </div>

                {result.percentage < 40 && (
                  <div className="rounded-2xl border border-amber-400/30 bg-amber-400/[0.06] px-5 py-3 text-xs font-mono text-amber-200">
                    <span className="font-semibold">Low-confidence reading.</span>{' '}
                    {numClasses} classes give a {randomBaseline}% random baseline. This {result.percentage}% result is close to that baseline, so treat the label as uncertain—not as a reliable identification.
                  </div>
                )}

                {/* Probability Spectrum Breakdown */}
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/[0.06] space-y-3">
                  <div className="font-mono text-[10px] tracking-wider uppercase text-zinc-500 flex justify-between">
                    <span>Probability Distribution Across {result.all_classes?.length || numClasses} Classes</span>
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
            ) : inferError ? (
              <div className="h-[280px] rounded-2xl bg-red-950/20 border border-red-500/30 flex flex-col items-center justify-center p-8 text-center text-red-300 font-mono text-xs space-y-3">
                <div className="w-10 h-10 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400">
                  <AlertTriangle className="w-5 h-5 animate-pulse" />
                </div>
                <div className="space-y-1">
                  <div className="text-red-400 font-sans font-semibold text-sm uppercase tracking-wider">Inference Tensor Error</div>
                  <div className="max-w-sm text-red-300/80 font-mono text-[11px] break-words">{inferError}</div>
                </div>
                <button
                  onClick={() => testMedia?.blob && handlePredict(testMedia.blob, testMedia.name)}
                  className="px-3.5 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/40 font-mono text-[11px] font-semibold tracking-wider transition-all"
                >
                  RETRY INFERENCE
                </button>
              </div>
            ) : (
              <div className="h-[280px] rounded-2xl bg-white/[0.02] border border-white/[0.06] flex flex-col items-center justify-center p-8 text-center text-zinc-500 font-mono text-xs space-y-2">
                <Cpu className="w-8 h-8 text-zinc-600" />
                <div className="text-zinc-300 font-sans font-semibold text-sm">Awaiting Specimen Input</div>
                <div className="max-w-sm text-zinc-500">
                  Select one of the sample presets above, drop an image/WAV, or click <strong className="text-cyan-400">Scan Live Webcam</strong> to execute real-time tensor inference directly through Apple Silicon MPS.
                </div>
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
