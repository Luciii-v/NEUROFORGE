// NEUROFORGE // Procedural Web Audio Sound Engine
// Synthesized in real time using the Web Audio API (Zero audio file dependencies).

class SoundEngine {
  constructor() {
    this.ctx = null;
    this.isMuted = false;
    this.masterGain = null;
    this.initialized = false;

    try {
      const saved = localStorage.getItem('neuroforge-sound-enabled');
      this.isMuted = saved !== null ? saved === 'false' : false;
    } catch {
      this.isMuted = false;
    }
  }

  init() {
    if (this.initialized && this.ctx) {
      if (this.ctx.state === 'suspended') {
        this.ctx.resume();
      }
      return;
    }

    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      this.ctx = new AudioCtx();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.setValueAtTime(this.isMuted ? 0 : 0.28, this.ctx.currentTime);
      this.masterGain.connect(this.ctx.destination);
      this.initialized = true;

      if (this.ctx.state === 'suspended') {
        const unlock = () => {
          if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
          }
          window.removeEventListener('pointerdown', unlock);
          window.removeEventListener('keydown', unlock);
        };
        window.addEventListener('pointerdown', unlock, { once: true });
        window.addEventListener('keydown', unlock, { once: true });
      }
    } catch (e) {
      console.warn('Web Audio initialization skipped:', e);
    }
  }

  setMuted(muted) {
    this.isMuted = muted;
    try {
      localStorage.setItem('neuroforge-sound-enabled', String(!muted));
    } catch {}

    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setValueAtTime(muted ? 0 : 0.28, this.ctx.currentTime);
    }
  }

  toggle() {
    this.init();
    const next = !this.isMuted;
    this.setMuted(next);
    if (!next) {
      this.tick();
    }
    return !next;
  }

  // Ultra-crisp micro tactile click (buttons, timeline scrubbers, node hovers)
  tick(freq = 1200, duration = 0.012) {
    if (this.isMuted) return;
    this.init();
    if (!this.ctx || this.ctx.state !== 'running') return;

    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    const filter = this.ctx.createBiquadFilter();

    filter.type = 'highpass';
    filter.frequency.setValueAtTime(800, t);

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(freq, t);
    osc.frequency.exponentialRampToValueAtTime(300, t + duration);

    gain.gain.setValueAtTime(0.12, t);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + duration);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);

    osc.start(t);
    osc.stop(t + duration);
  }

  // Soft node hover
  hover() {
    this.tick(1600, 0.008);
  }

  // Synaptic Light Pulse (Forward pass through layer)
  synapsePulse(layerIdx = 0, totalLayers = 4) {
    if (this.isMuted) return;
    this.init();
    if (!this.ctx || this.ctx.state !== 'running') return;

    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    // Ascending pentatonic pitch based on layer index: D4, F#4, A4, D5, F#5
    const pitches = [293.66, 369.99, 440.0, 587.33, 739.99];
    const baseFreq = pitches[layerIdx % pitches.length] || 440;

    osc.type = 'sine';
    osc.frequency.setValueAtTime(baseFreq * 0.9, t);
    osc.frequency.exponentialRampToValueAtTime(baseFreq * 1.3, t + 0.08);

    gain.gain.setValueAtTime(0.16, t);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.1);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(t);
    osc.stop(t + 0.1);
  }

  // Tensor Forward Pass Burst (Fast multi-tone sweep on inference start)
  inferStart() {
    if (this.isMuted) return;
    this.init();
    if (!this.ctx || this.ctx.state !== 'running') return;

    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    const filter = this.ctx.createBiquadFilter();

    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(1400, t);
    filter.Q.setValueAtTime(3, t);

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(320, t);
    osc.frequency.exponentialRampToValueAtTime(960, t + 0.18);

    gain.gain.setValueAtTime(0.14, t);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.2);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);

    osc.start(t);
    osc.stop(t + 0.2);
  }

  // Triumphal harmonic chord on classification resolution or high confidence
  championChord(highConfidence = true) {
    if (this.isMuted) return;
    this.init();
    if (!this.ctx || this.ctx.state !== 'running') return;

    const t = this.ctx.currentTime;
    // Lush open chord: E4, B4, E5, G#5
    const freqs = highConfidence ? [329.63, 493.88, 659.25, 830.61] : [261.63, 329.63, 392.0];

    freqs.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, t);

      const delay = idx * 0.025;
      gain.gain.setValueAtTime(0, t);
      gain.gain.setValueAtTime(0.12 / freqs.length, t + delay);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + delay + 0.65);

      osc.connect(gain);
      gain.connect(this.masterGain);

      osc.start(t + delay);
      osc.stop(t + delay + 0.7);
    });
  }

  // Genetic mutation blip
  mutate() {
    if (this.isMuted) return;
    this.init();
    if (!this.ctx || this.ctx.state !== 'running') return;

    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'square';
    osc.frequency.setValueAtTime(440, t);
    osc.frequency.setValueAtTime(660, t + 0.04);
    osc.frequency.setValueAtTime(880, t + 0.08);

    gain.gain.setValueAtTime(0.06, t);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.14);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(t);
    osc.stop(t + 0.15);
  }
}

export const soundEngine = new SoundEngine();
export default soundEngine;
