import {Activity, Check} from 'lucide-react';

import type {Mode} from '../types';


export default function ProcessingState({mode}: {mode: Mode}) {
  const steps =
    mode === 'fast'
      ? ['Grounding', 'Generating', 'Executing']
      : ['Grounding', 'Generating 8 candidates', 'Executing', 'Voting'];

  return (
    <div className="processing">
      <div className="processing-label">
        <Activity size={15} />
        Processing query
      </div>
      <div className="steps">
        {steps.map((step, index) => (
          <div className="step" key={step}>
            <span className="step-dot">
              {index === 0 ? <Check size={12} /> : index + 1}
            </span>
            <span>{step}</span>
            {index < steps.length - 1 && <span className="step-line" />}
          </div>
        ))}
      </div>
    </div>
  );
}
