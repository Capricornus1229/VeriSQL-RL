import type {QueryResponse} from '../types';


export default function CandidateResult({response}: {response: QueryResponse}) {
  return (
    <div className="candidate-list">
      {response.candidates.map((candidate) => (
        <details
          className={candidate.selected ? 'candidate selected' : 'candidate'}
          key={candidate.index}
        >
          <summary>
            <b>#{candidate.index}</b>
            <span>{candidate.kind}</span>
            <span className={candidate.execution_status === 'success' ? 'success' : 'failure'}>
              {candidate.execution_status}
            </span>
            {candidate.execution_status === 'success' && (
              <span>{candidate.row_count} rows · {candidate.execution_elapsed_ms.toFixed(1)} ms</span>
            )}
            {candidate.cluster_support > 0 && (
              <span>support {candidate.cluster_support}</span>
            )}
            {candidate.mean_logprob !== null && (
              <span>log p {candidate.mean_logprob.toFixed(3)}</span>
            )}
            {candidate.selected && <em>Vote selected</em>}
          </summary>
          {candidate.reasoning && (
            <p className="candidate-reasoning">{candidate.reasoning}</p>
          )}
          <pre>{candidate.sql || candidate.raw_output}</pre>
        </details>
      ))}
    </div>
  );
}
