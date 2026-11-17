import React, { useEffect, useState } from 'react';

const stages = [
  { number: '01', word: 'BUILD', line: 'assemble the first blueprint' },
  { number: '02', word: 'MUTATE', line: 'let the architecture change' },
  { number: '03', word: 'SHIP', line: 'keep the evidence with the model' },
];

export default function CinematicSplash({ onComplete }) {
  const [activeStage, setActiveStage] = useState(0);

  useEffect(() => {
    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const duration = reducedMotion ? 650 : 5600;
    const stageTimer = setInterval(() => {
      setActiveStage((current) => Math.min(current + 1, stages.length - 1));
    }, reducedMotion ? 160 : 1200);
    const closeTimer = setTimeout(() => onComplete?.(), duration);
    return () => {
      clearInterval(stageTimer);
      clearTimeout(closeTimer);
    };
  }, [onComplete]);

  return (
    <div className="nf-opening" role="dialog" aria-label="NEUROFORGE introduction">
      <div className="nf-opening-grid" aria-hidden="true" />
      <div className="nf-opening-noise" aria-hidden="true" />
      <div className="nf-opening-rail nf-opening-rail-one" aria-hidden="true" />
      <div className="nf-opening-rail nf-opening-rail-two" aria-hidden="true" />

      <button className="nf-opening-skip" onClick={onComplete}>Skip intro</button>

      <div className="nf-opening-content">
        <div className="nf-opening-kicker"><span /> a living architecture search process</div>
        <div className="nf-opening-wordmark" aria-label="NEUROFORGE">
          {Array.from('NEUROFORGE').map((letter, index) => (
            <span key={`${letter}-${index}`} style={{ '--letter-index': index }}>{letter}</span>
          ))}
        </div>
        <div className="nf-opening-title">Build.<i> Mutate.</i> Ship.</div>

        <div className="nf-opening-loader" aria-label="Initializing NEUROFORGE workspace">
          <div className="nf-opening-loader-meta">
            <span>initializing the workspace</span>
            <span>NEUROFORGE / readying instruments</span>
          </div>
          <div className="nf-opening-loader-track"><span /></div>
        </div>

        <div className="nf-opening-stage-rail" aria-label="Build mutate ship sequence">
          {stages.map((stage, index) => (
            <div className={`nf-opening-stage ${index <= activeStage ? 'is-active' : ''}`} key={stage.word}>
              <span className="nf-opening-stage-number">{stage.number}</span>
              <strong>{stage.word}</strong>
              <small>{stage.line}</small>
            </div>
          ))}
        </div>

        <div className="nf-opening-specimen" aria-hidden="true">
          <div className="nf-opening-layer nf-opening-layer-input"><span>input</span></div>
          <div className="nf-opening-wire wire-a" />
          <div className="nf-opening-layer nf-opening-layer-conv"><span>conv</span></div>
          <div className="nf-opening-wire wire-b" />
          <div className="nf-opening-layer nf-opening-layer-mutated"><span>mutation</span></div>
          <div className="nf-opening-wire wire-c" />
          <div className="nf-opening-layer nf-opening-layer-head"><span>ship</span></div>
          <span className="nf-opening-pulse pulse-a" />
          <span className="nf-opening-pulse pulse-b" />
          <span className="nf-opening-pulse pulse-c" />
        </div>

        <div className="nf-opening-footer">
          <span>NEUROFORGE / MODEL DEVELOPMENT STUDIO</span>
          <span>entering the production workspace</span>
        </div>
      </div>
    </div>
  );
}
