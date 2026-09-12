import {useState} from 'react';
import {AnimatePresence, motion} from 'motion/react';
import {ChevronDown, Search, Table2, X} from 'lucide-react';

import type {DatabaseDetail, DatabaseSummary} from '../types';


type Props = {
  db: DatabaseDetail | null;
  dbs: DatabaseSummary[];
  onSelect: (dbId: string) => void;
  onClose: () => void;
};


export default function DatabaseDrawer({db, dbs, onSelect, onClose}: Props) {
  const [databaseQuery, setDatabaseQuery] = useState('');
  const [schemaQuery, setSchemaQuery] = useState('');
  const normalizedSchemaQuery = schemaQuery.toLowerCase();
  const tables = (db?.tables ?? []).filter((table) =>
    [table.name, table.display_name, ...table.columns.flatMap((column) => [
      column.name,
      column.display_name,
      column.type,
    ])]
      .join(' ')
      .toLowerCase()
      .includes(normalizedSchemaQuery),
  );

  return (
    <AnimatePresence>
      <motion.div
        className="drawer-backdrop"
        initial={{opacity: 0}}
        animate={{opacity: 1}}
        exit={{opacity: 0}}
        onClick={onClose}
      >
        <motion.section
          className="drawer"
          initial={{x: '100%'}}
          animate={{x: 0}}
          exit={{x: '100%'}}
          transition={{duration: 0.22}}
          onClick={(event) => event.stopPropagation()}
        >
          <div className="drawer-head">
            <div>
              <p className="eyebrow">DATABASE &amp; SCHEMA EXPLORER</p>
              <h2>{db?.db_id ?? 'Select a database'}</h2>
            </div>
            <button className="icon-btn" onClick={onClose} aria-label="Close">
              <X />
            </button>
          </div>

          <div className="drawer-database-picker">
            <div className="search">
              <Search size={15} />
              <input
                placeholder="Search databases"
                value={databaseQuery}
                onChange={(event) => setDatabaseQuery(event.target.value)}
              />
            </div>
            <div className="drawer-db-list">
              {dbs
                .filter((item) =>
                  item.db_id.toLowerCase().includes(databaseQuery.toLowerCase()),
                )
                .map((item) => (
                  <button
                    className={item.db_id === db?.db_id ? 'active' : ''}
                    key={item.db_id}
                    onClick={() => onSelect(item.db_id)}
                  >
                    <span>{item.db_id}</span>
                    <small>{item.split} · {item.table_count} tables</small>
                  </button>
                ))}
            </div>
          </div>

          <div className="search schema-search">
            <Search size={15} />
            <input
              placeholder="Search tables or columns"
              value={schemaQuery}
              onChange={(event) => setSchemaQuery(event.target.value)}
            />
          </div>
          <div className="tables">
            {tables.map((table) => (
              <details key={table.name} open>
                <summary>
                  <Table2 size={14} />
                  {table.name}
                  <ChevronDown size={14} />
                </summary>
                <div className="columns">
                  {table.columns.map((column) => (
                    <div key={column.name}>
                      <span>{column.name}</span>
                      <small>{column.type.toUpperCase()}</small>
                      {table.primary_keys.includes(column.name) && <em>PK</em>}
                    </div>
                  ))}
                </div>
              </details>
            ))}
            {(db?.foreign_keys.length ?? 0) > 0 && (
              <div className="foreign-keys">
                <p className="eyebrow">FOREIGN KEYS</p>
                {db?.foreign_keys.map((key) => (
                  <code key={`${key.source_table}.${key.source_column}`}>
                    {key.source_table}.{key.source_column} → {key.target_table}.
                    {key.target_column}
                  </code>
                ))}
              </div>
            )}
          </div>
        </motion.section>
      </motion.div>
    </AnimatePresence>
  );
}
