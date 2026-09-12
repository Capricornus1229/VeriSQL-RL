import {ChevronDown, Play} from 'lucide-react';

import type {Mode} from '../types';
import ModeSwitch from './ModeSwitch';


type Props = {
  question: string;
  setQuestion: (value: string) => void;
  evidence: string;
  setEvidence: (value: string) => void;
  showEvidence: boolean;
  setShowEvidence: (value: boolean) => void;
  mode: Mode;
  setMode: (value: Mode) => void;
  onRun: () => void;
  loading: boolean;
  disabled: boolean;
};


export default function QueryComposer({
  question,
  setQuestion,
  evidence,
  setEvidence,
  showEvidence,
  setShowEvidence,
  mode,
  setMode,
  onRun,
  loading,
  disabled,
}: Props) {
  return (
    <section className="composer">
      <textarea
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        placeholder="e.g. Which district has the highest average salary?"
        maxLength={1000}
      />
      <div className="composer-foot">
        <button
          className="evidence-toggle"
          onClick={() => setShowEvidence(!showEvidence)}
        >
          <ChevronDown size={15} />
          Evidence <small>optional</small>
        </button>
        <span className="chars">{question.length}/1000</span>
      </div>
      {showEvidence && (
        <textarea
          className="evidence"
          value={evidence}
          onChange={(event) => setEvidence(event.target.value)}
          placeholder="Add hints about schema terms, values, or business definitions…"
        />
      )}
      <div className="composer-actions">
        <ModeSwitch mode={mode} setMode={setMode} />
        <button
          className="run"
          disabled={disabled || loading}
          onClick={onRun}
        >
          {loading ? (
            'Running…'
          ) : (
            <>
              <Play size={15} fill="currentColor" />
              Run Query
            </>
          )}
        </button>
      </div>
    </section>
  );
}
