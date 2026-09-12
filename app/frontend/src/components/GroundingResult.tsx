import type {QueryResponse} from '../types';


export default function GroundingResult({response}: {response: QueryResponse}) {
  return (
    <div className="result-pane grounding">
      {response.grounding.length === 0 && (
        <p className="muted">No additional database values were retrieved.</p>
      )}
      {response.grounding.map((hit) => (
        <div className="ground" key={`${hit.table}.${hit.column}`}>
          <b>{hit.table}.{hit.column}</b>
          {hit.description && <span>{hit.description}</span>}
          <div>
            {hit.matched_values.map((value) => (
              <em key={String(value)}>{String(value)}</em>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
