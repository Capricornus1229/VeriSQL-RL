import type {QueryResponse} from '../types';


export default function ReasoningResult({response}: {response: QueryResponse}) {
  return (
    <div className="result-pane">
      <details>
        <summary>
          Query Plan <small>Model-generated planning trace</small>
        </summary>
        <p className="reasoning">
          {response.selected_candidate?.reasoning || 'No planning trace returned.'}
        </p>
      </details>
    </div>
  );
}
