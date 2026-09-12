import {Check, Clipboard} from 'lucide-react';

import type {QueryResponse} from '../types';


export default function SqlResult({response}: {response: QueryResponse}) {
  const candidate = response.selected_candidate;
  const execution = response.execution;
  return (
    <div className="result-pane">
      <div className="result-meta">
        <span className={execution?.status === 'success' ? 'success' : 'failure'}>
          <Check size={14} />
          {execution?.status ?? 'No executable SQL'}
        </span>
        <span>
          {candidate?.format_compliance ? 'Format compliant' : candidate?.extraction_status ?? 'Extraction failed'}
        </span>
      </div>
      <div className="code-head">
        <span>Generated SQL</span>
        <button onClick={() => navigator.clipboard.writeText(candidate?.sql ?? '')}>
          <Clipboard size={14} />Copy
        </button>
      </div>
      <pre><code>{candidate?.sql || 'No SQL was extracted.'}</code></pre>
      <div className="access-chips">
        {execution?.accessed_tables.map((table) => <em key={table}>{table}</em>)}
        {execution?.accessed_columns.map((item) => (
          <em key={`${item.table}.${item.column}`}>{item.table}.{item.column}</em>
        ))}
      </div>
    </div>
  );
}
