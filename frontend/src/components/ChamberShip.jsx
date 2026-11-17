import React, { useState, useRef } from 'react';
import { Rocket, CheckCircle2, AlertCircle, UploadCloud, Activity, Zap, HardDrive, BarChart3, RotateCcw } from 'lucide-react';

export default function ChamberShip({ checkpoint, onSetSpeech, onModelRetrained }) {
  const [trainingEpochs, setTrainingEpochs] = useState(15);
  const [training, setTraining] = useState(false);
  const [testImage, setTestImage] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [trainStatus, setTrainStatus] = useState(null);
  const fileInputRef = useRef(null);

  const handleDeepTrain = async () => {
    if (training) return;
    setTraining(true);
    setTrainStatus('Initiating full training cycle across all epochs...');
    if (onSetSpeech) onSetSpeech(`Feeding champion DNA into the furnace for deep training (${trainingEpochs} epochs)...`, 'forging');

    try {
      const res = await fetch('/api/train/full', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs: Number(trainingEpochs) }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Training failed');

      setTrainStatus('Full training completed successfully! New weights forged.');
      if (onSetSpeech) onSetSpeech(`Deep training complete! Checkpoint saved and verified on held-out test split. Ready for live testing!`, 'victory');
      if (onModelRetrained) onModelRetrained(data.metrics);
    } catch (err) {
      setTrainStatus(`Error: ${err.message}`);
      if (onSetSpeech) onSetSpeech(`Furnace error during deep training: ${err.message}`, 'caution');
    } finally {
      setTraining(false);
    }
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setTestImage(URL.createObjectURL(file));
    setPredicting(true);
    setPredictionResult(null);
    if (onSetSpeech) onSetSpeech(`Passing sample '${file.name}' into the Crucible for inference...`, 'forging');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Inference error');

      setPredictionResult(data);
      if (onSetSpeech) onSetSpeech(`Prediction complete! Identified as '${data.prediction}' with ${data.percentage}% confidence in ${data.latency_ms}ms.`, 'victory');
    } catch (err) {
      if (onSetSpeech) onSetSpeech(`Inference failed: ${err.message}`, 'caution');
    } finally {
      setPredicting(false);
    }
  };

  const metrics = checkpoint?.best_test_metrics || {};

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-emerald-500/10 pb-4">
        <h2 className="font-cinzel text-xl md:text-2xl font-bold tracking-wide text-slate-100 flex items-center gap-3">
          <Rocket className="w-6 h-6 text-emerald-400" />
          <span>CHAMBER III: SHIP & LIVE CRUCIBLE</span>
        </h2>
        <p className="text-slate-400 text-sm mt-1">
          Finalize model weights with full-epoch training, inspect test-split metrics, and test single-image inference in real time.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Full Train & Metrics */}
        <div className="lg:col-span-6 space-y-5">
          {/* Deep Train Panel */}
          <div className="atelier-panel rounded-xl p-5 border-emerald-500/20">
            <h3 className="font-cinzel text-sm font-bold text-emerald-300 uppercase tracking-wider mb-2 flex items-center gap-2">
              <Zap className="w-4 h-4 text-emerald-400" />
              DEEP TRAINING CONSOLE
            </h3>
            <p className="text-slate-400 text-xs mb-4">
              Train the selected champion architecture on the full training split with augmented transforms and class weighting.
            </p>

            <div className="flex items-center gap-3 mb-4">
              <div className="flex-1">
                <label className="block font-mono text-[10px] text-slate-400 uppercase tracking-wider mb-1">
                  Epochs
                </label>
                <input
                  type="number"
                  min="5"
                  max="50"
                  value={trainingEpochs}
                  disabled={training}
                  onChange={(e) => setTrainingEpochs(e.target.value)}
                  className="w-full bg-black/60 border border-emerald-500/20 rounded-lg px-3 py-2 text-sm font-mono text-slate-200"
                />
              </div>

              <div className="flex-1 pt-4">
                <button
                  onClick={handleDeepTrain}
                  disabled={training}
                  className="btn-alchemist w-full py-2.5 rounded-lg text-xs font-bold flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {training ? (
                    <>
                      <RotateCcw className="w-4 h-4 animate-spin text-amber-950" />
                      <span>FORGING...</span>
                    </>
                  ) : (
                    <>
                      <Rocket className="w-4 h-4" />
                      <span>COMMENCE TRAINING</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {trainStatus && (
              <div className={`p-3 rounded-lg text-xs font-mono border ${trainStatus.startsWith('Error') ? 'bg-rose-500/10 border-rose-500/30 text-rose-300' : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'}`}>
                {trainStatus}
              </div>
            )}
          </div>

          {/* Held-Out Metrics Display */}
          <div className="atelier-panel rounded-xl p-5 border-white/5">
            <h3 className="font-cinzel text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-amber-400" />
              HELD-OUT TEST VERIFICATION
            </h3>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-center">
              <div className="p-3 rounded-lg bg-black/50 border border-white/5">
                <div className="text-[10px] text-slate-400 uppercase">Accuracy</div>
                <div className="text-base font-bold text-emerald-400 mt-1">
                  {metrics.accuracy ? `${metrics.accuracy.toFixed(1)}%` : '—'}
                </div>
              </div>

              <div className="p-3 rounded-lg bg-black/50 border border-white/5">
                <div className="text-[10px] text-slate-400 uppercase">Macro F1</div>
                <div className="text-base font-bold text-purple-400 mt-1">
                  {metrics.f1 ? `${metrics.f1.toFixed(1)}%` : '—'}
                </div>
              </div>

              <div className="p-3 rounded-lg bg-black/50 border border-white/5">
                <div className="text-[10px] text-slate-400 uppercase">Precision</div>
                <div className="text-base font-bold text-cyan-400 mt-1">
                  {metrics.precision ? `${metrics.precision.toFixed(1)}%` : '—'}
                </div>
              </div>

              <div className="p-3 rounded-lg bg-black/50 border border-white/5">
                <div className="text-[10px] text-slate-400 uppercase">Latency/Sample</div>
                <div className="text-base font-bold text-amber-400 mt-1">
                  {metrics.inference_latency_per_sample_seconds ? `${(metrics.inference_latency_per_sample_seconds * 1000).toFixed(1)}ms` : '—'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Live Testing Crucible */}
        <div className="lg:col-span-6">
          <div className="atelier-panel rounded-xl p-5 border-amber-500/20 h-full flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-amber-500/10 mb-4">
                <span className="font-cinzel text-xs text-amber-300 font-bold uppercase tracking-wider flex items-center gap-2">
                  <Activity className="w-4 h-4 text-amber-400" />
                  LIVE TESTING CRUCIBLE
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                  {checkpoint?.ready ? 'MODEL ACTIVE' : 'CHECKPOINT PENDING'}
                </span>
              </div>

              {/* Upload Dropzone */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className="atelier-panel atelier-panel-hover rounded-xl p-6 border-dashed border-2 border-amber-500/30 text-center cursor-pointer mb-5 group"
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
                {testImage ? (
                  <div className="flex flex-col items-center">
                    <img src={testImage} alt="Test preview" className="w-28 h-28 object-cover rounded-lg border border-amber-500/40 mb-2 shadow-lg" />
                    <span className="text-[11px] font-mono text-slate-300">Click to test a different image</span>
                  </div>
                ) : (
                  <div>
                    <UploadCloud className="w-8 h-8 text-amber-400 mx-auto mb-2 group-hover:scale-110 transition-transform" />
                    <div className="font-cinzel text-xs font-bold text-slate-200 uppercase">Drop Single Test Image</div>
                    <div className="text-[10px] font-mono text-slate-500 mt-0.5">PNG, JPG, or WEBP</div>
                  </div>
                )}
              </div>

              {/* Prediction Result & Vacuum Gauges */}
              {predicting && (
                <div className="text-center py-6 text-amber-400 font-mono text-xs animate-pulse">
                  Crucible evaluating image features...
                </div>
              )}

              {predictionResult && !predicting && (
                <div className="space-y-4 font-mono text-xs">
                  <div className="p-4 rounded-xl bg-black/60 border border-amber-500/30 flex items-center justify-between">
                    <div>
                      <div className="text-[10px] text-slate-400 uppercase">Top Prediction</div>
                      <div className="text-base font-bold text-amber-300 font-cinzel mt-0.5">
                        {predictionResult.prediction}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-base font-bold text-emerald-400">
                        {predictionResult.percentage}%
                      </div>
                      <div className="text-[10px] text-slate-400">
                        {predictionResult.latency_ms}ms latency
                      </div>
                    </div>
                  </div>

                  {/* Multi-class Vacuum Gauges */}
                  <div className="space-y-2">
                    <span className="text-[11px] text-slate-400 block mb-1">Alchemical Distribution:</span>
                    {predictionResult.all_classes?.map((item, idx) => (
                      <div key={idx} className="space-y-1">
                        <div className="flex justify-between text-[11px]">
                          <span className={idx === 0 ? 'text-amber-300 font-bold' : 'text-slate-400'}>
                            {item.class_name}
                          </span>
                          <span className="text-slate-300">{item.percentage}%</span>
                        </div>
                        <div className="h-2 rounded-full bg-black/80 border border-white/5 overflow-hidden p-[1px]">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              idx === 0
                                ? 'bg-gradient-to-r from-amber-500 to-amber-300 shadow-[0_0_10px_#f59e0b]'
                                : 'bg-slate-700'
                            }`}
                            style={{ width: `${item.percentage}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span>Ready for Export</span>
              <span className="text-emerald-400 font-bold">ARTIFACTS AVAILABLE</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
