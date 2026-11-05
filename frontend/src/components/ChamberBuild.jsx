import React, { useState } from 'react';
import { Database, FolderCheck, UploadCloud, Layers, ShieldCheck, Sparkles } from 'lucide-react';

export default function ChamberBuild({ onDatasetLoaded, currentDataset, onSetSpeech }) {
  const [localPath, setLocalPath] = useState('/Users/mahathavivaanveer/Developer/NEUROFORGE/custom_data/images/crop pest dataset');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleLoadPath = async () => {
    if (!localPath.trim()) return;
    setLoading(true);
    setError(null);
    if (onSetSpeech) onSetSpeech('Inspecting your directory structure and measuring class distribution...', 'curious');

    try {
      const res = await fetch('/api/dataset/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: localPath }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to inspect dataset path');

      if (onDatasetLoaded) onDatasetLoaded(data.metadata);
      if (onSetSpeech) onSetSpeech(`Excellent! Discovered ${data.metadata.num_classes} classes and ${data.metadata.num_samples} samples. Ready for genetic mutation.`, 'victory');
    } catch (err) {
      setError(err.message);
      if (onSetSpeech) onSetSpeech(`Hmm, couldn't read that folder: ${err.message}`, 'caution');
    } finally {
      setLoading(false);
    }
  };

  const handleZipUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    if (onSetSpeech) onSetSpeech(`Unpacking archive '${file.name}' into the local workspace...`, 'curious');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/dataset/upload-zip', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to unpack ZIP');

      if (onDatasetLoaded) onDatasetLoaded(data.metadata);
      if (onSetSpeech) onSetSpeech(`Archive unpacked! ${data.metadata.num_classes} classes ready for the furnace.`, 'victory');
    } catch (err) {
      setError(err.message);
      if (onSetSpeech) onSetSpeech(`Failed to unpack archive: ${err.message}`, 'caution');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Chamber Header */}
      <div className="border-b border-amber-500/10 pb-4">
        <h2 className="font-cinzel text-xl md:text-2xl font-bold tracking-wide text-slate-100 flex items-center gap-3">
          <Database className="w-6 h-6 text-amber-400" />
          <span>CHAMBER I: THE FOUNDRY INTAKE</span>
        </h2>
        <p className="text-slate-400 text-sm mt-1">
          Ingest image folders, apply class-frequency balancing, and calibrate train/validation partitions.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Intake Controls */}
        <div className="lg:col-span-7 space-y-5">
          {/* Path Loader */}
          <div className="atelier-panel rounded-xl p-5 border-amber-500/20">
            <label className="block font-mono text-xs text-amber-300 font-bold uppercase tracking-wider mb-2">
              Absolute Local Dataset Path
            </label>
            <div className="flex gap-3">
              <input
                type="text"
                value={localPath}
                onChange={(e) => setLocalPath(e.target.value)}
                placeholder="/path/to/dataset"
                className="flex-1 bg-black/60 border border-amber-500/20 rounded-lg px-4 py-2.5 text-sm font-mono text-slate-200 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400"
              />
              <button
                onClick={handleLoadPath}
                disabled={loading}
                className="btn-alchemist px-5 py-2.5 rounded-lg text-xs font-bold shrink-0 disabled:opacity-50"
              >
                {loading ? 'INSPECTING...' : 'BIND PATH'}
              </button>
            </div>
          </div>

          {/* ZIP Archive Dropzone */}
          <div className="atelier-panel rounded-xl p-6 border-dashed border-2 border-amber-500/30 text-center relative group hover:border-amber-400 transition-colors">
            <input
              type="file"
              accept=".zip"
              onChange={handleZipUpload}
              disabled={loading}
              className="absolute inset-0 opacity-0 cursor-pointer disabled:cursor-not-allowed"
            />
            <UploadCloud className="w-10 h-10 text-amber-400/80 mx-auto mb-3 group-hover:scale-110 transition-transform" />
            <h4 className="font-cinzel text-sm font-bold text-slate-200 uppercase tracking-wider">
              Drop Dataset ZIP Archive
            </h4>
            <p className="text-slate-400 text-xs mt-1">
              Supports standard ImageFolder ZIPs or Roboflow single-label exports
            </p>
          </div>

          {error && (
            <div className="p-4 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs font-mono">
              {error}
            </div>
          )}
        </div>

        {/* Right Column: Dataset Spec Card */}
        <div className="lg:col-span-5">
          <div className="atelier-panel rounded-xl p-5 border-amber-500/20 h-full flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-amber-500/10 mb-4">
                <span className="font-mono text-xs text-amber-300 font-bold uppercase tracking-wider flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  CALIBRATED SPECIFICATION
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">
                  {currentDataset?.format || 'AWAITING BIND'}
                </span>
              </div>

              {currentDataset ? (
                <div className="space-y-4 font-mono text-xs">
                  <div className="flex justify-between items-center py-1 border-b border-white/5">
                    <span className="text-slate-400">Class Count:</span>
                    <span className="text-amber-200 font-bold">{currentDataset.num_classes}</span>
                  </div>
                  <div className="flex justify-between items-center py-1 border-b border-white/5">
                    <span className="text-slate-400">Total Samples:</span>
                    <span className="text-amber-200 font-bold">{currentDataset.num_samples || '—'}</span>
                  </div>
                  <div className="flex justify-between items-center py-1 border-b border-white/5">
                    <span className="text-slate-400">Balancing Mode:</span>
                    <span className="text-emerald-400 font-bold">Inverse Freq. Weighting</span>
                  </div>

                  <div>
                    <span className="text-slate-400 block mb-2">Detected Classes:</span>
                    <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto pr-1">
                      {currentDataset.class_names?.map((name, i) => (
                        <span key={i} className="px-2 py-0.5 rounded bg-black/60 border border-amber-500/20 text-slate-300 text-[11px]">
                          {name}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-slate-500 text-xs font-mono">
                  No dataset bound yet. Provide an absolute directory or drop a ZIP to activate.
                </div>
              )}
            </div>

            <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span>Ready for Phase II</span>
              <span className="text-amber-400 font-bold">MUTATE CHAMBER →</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
