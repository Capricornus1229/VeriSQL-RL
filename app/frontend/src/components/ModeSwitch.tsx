import {motion} from 'motion/react';

import type {Mode} from '../types';


type Props = {
  mode: Mode;
  setMode: (mode: Mode) => void;
};


export default function ModeSwitch({mode, setMode}: Props) {
  return (
    <div className="mode-wrap">
      <div className="mode-switch">
        {(['fast', 'accurate'] as Mode[]).map((item) => (
          <button
            key={item}
            onClick={() => setMode(item)}
            className={mode === item ? 'chosen' : ''}
          >
            {mode === item && (
              <motion.span layoutId="mode-pill" className="mode-pill" />
            )}
            <span>{item[0].toUpperCase() + item.slice(1)}</span>
          </button>
        ))}
      </div>
      <p>
        {mode === 'fast'
          ? '1 candidate · lower latency'
          : '8 candidates · execution vote'}
      </p>
    </div>
  );
}
