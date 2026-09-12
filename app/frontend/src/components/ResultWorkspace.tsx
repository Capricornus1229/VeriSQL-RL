import {useState} from 'react';
import {AnimatePresence, motion} from 'motion/react';

import type {Mode, QueryResponse} from '../types';
import CandidateResult from './CandidateResult';
import DataResult from './DataResult';
import GroundingResult from './GroundingResult';
import ReasoningResult from './ReasoningResult';
import SqlResult from './SqlResult';


export default function ResultWorkspace({response, mode}: {response: QueryResponse; mode: Mode}) {
  const [tab, setTab] = useState('Data');
  const tabs = ['Data', 'SQL', 'Query Plan', 'Grounding', ...(mode === 'accurate' ? ['Candidates'] : [])];
  return (
    <section className="results">
      <div className="result-summary">
        <span>{response.status.replace(/_/g, ' ')}</span>
        <span>{response.timings.total_ms.toFixed(0)} ms total</span>
        <span>grounding {response.timings.grounding_ms.toFixed(0)} ms</span>
        <span>generation {response.timings.generation_ms.toFixed(0)} ms</span>
        <span>
          execution {(response.timings.candidate_execution_ms + response.timings.display_execution_ms).toFixed(0)} ms
        </span>
        {response.vote && <span>{response.vote.support} vote support</span>}
      </div>
      <div className="tabs">
        {tabs.map((item) => (
          <button
            className={tab === item ? 'tab-active' : ''}
            onClick={() => setTab(item)}
            key={item}
          >
            {item}
          </button>
        ))}
      </div>
      <AnimatePresence mode="wait">
        <motion.div
          key={tab}
          initial={{opacity: 0, y: 4}}
          animate={{opacity: 1, y: 0}}
          exit={{opacity: 0, y: -4}}
          transition={{duration: 0.18}}
        >
          {tab === 'Data' && <DataResult response={response} />}
          {tab === 'SQL' && <SqlResult response={response} />}
          {tab === 'Query Plan' && <ReasoningResult response={response} />}
          {tab === 'Grounding' && <GroundingResult response={response} />}
          {tab === 'Candidates' && <CandidateResult response={response} />}
        </motion.div>
      </AnimatePresence>
    </section>
  );
}
