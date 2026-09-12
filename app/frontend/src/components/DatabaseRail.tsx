import {useState} from 'react';
import {Database, PanelRight, Search} from 'lucide-react';

import type {DatabaseSummary} from '../types';


type Props = {
  dbs: DatabaseSummary[];
  selected: string;
  onSelect: (database: DatabaseSummary) => void;
  onSchema: () => void;
  onMobile: () => void;
  onExample: (question: string, evidence: string) => void;
};


export default function DatabaseRail({
  dbs,
  selected,
  onSelect,
  onSchema,
  onMobile,
  onExample,
}: Props) {
  const [query, setQuery] = useState('');
  const selectedDatabase = dbs.find((item) => item.db_id === selected);
  const visible = dbs.filter((item) =>
    item.db_id.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <>
      <aside className="rail">
        <div className="rail-title">
          <span>Databases</span><span className="count">{dbs.length || '—'}</span>
        </div>
        <div className="search">
          <Search size={15} />
          <input
            placeholder="Search databases"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        <div className="db-list">
          {visible.map((database) => (
            <button
              className={`db-row ${selected === database.db_id ? 'active' : ''}`}
              onClick={() => onSelect(database)}
              key={database.db_id}
            >
              <Database size={16} />
              <span className="db-name">{database.db_id}</span>
              <span className="db-meta">
                {database.split} · {database.table_count} tables
              </span>
            </button>
          ))}
        </div>
        {(selectedDatabase?.example_queries.length ?? 0) > 0 && (
          <div className="rail-examples">
            <span>Example questions</span>
            {selectedDatabase?.example_queries.map((example) => (
              <button
                key={example.question}
                onClick={() => onExample(example.question, example.evidence)}
              >
                {example.question}
              </button>
            ))}
          </div>
        )}
        <button className="schema-btn" onClick={onSchema}>
          <PanelRight size={15} />View Schema
        </button>
      </aside>
      <button className="mobile-db" onClick={onMobile}>Browse databases</button>
    </>
  );
}
