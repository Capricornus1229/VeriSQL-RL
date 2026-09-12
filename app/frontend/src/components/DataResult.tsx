import {Check, Clipboard, Download} from 'lucide-react';

import type {QueryResponse} from '../types';
import EmptyState from './EmptyState';


export default function DataResult({response}: {response: QueryResponse}) {
  const execution = response.execution;
  if (!execution) {
    return (
      <EmptyState
        title="No executable candidate"
        message="The model did not produce a SQL query that SQLite could execute."
      />
    );
  }
  if (execution.status !== 'success') {
    return (
      <EmptyState
        title="SQL execution failed"
        message={execution.error_message ?? 'The selected SQL could not be executed.'}
      />
    );
  }

  const rows = execution.rows ?? [];
  const objects = rows.map((row) =>
    Object.fromEntries(execution.columns.map((column, index) => [column, row[index]])),
  );
  const downloadCsv = () => {
    const csv = [execution.columns, ...rows]
      .map((row) =>
        row
          .map((value) => `"${String(value ?? '').replace(/"/g, '""')}"`)
          .join(','),
      )
      .join('\n');
    const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'verisql-results.csv';
    anchor.click();
    URL.revokeObjectURL(url);
  };

  if (!rows.length) {
    return <EmptyState />;
  }
  return (
    <div className="result-pane">
      <div className="result-meta">
        <span className="success"><Check size={14} />Executed successfully</span>
        <span>{execution.row_count} rows</span>
        <span>{execution.elapsed_ms.toFixed(2)} ms</span>
        {execution.result_truncated && (
          <span>Showing first {execution.displayed_row_count} rows</span>
        )}
        <button onClick={() => navigator.clipboard.writeText(JSON.stringify(objects, null, 2))}>
          <Clipboard size={14} />Copy JSON
        </button>
        <button onClick={downloadCsv}><Download size={14} />Download CSV</button>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>{execution.columns.map((column) => <th key={column}>{column}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {execution.columns.map((column, columnIndex) => (
                  <td key={`${column}-${columnIndex}`}>
                    {String(row[columnIndex] ?? 'NULL')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
