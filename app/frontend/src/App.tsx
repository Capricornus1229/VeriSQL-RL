import {useEffect, useMemo, useState} from 'react';
import {Database, Zap} from 'lucide-react';

import {getDatabase, getDatabases, getHealth} from './api/client';
import type {
  DatabaseDetail,
  DatabaseSummary,
  HealthResponse,
  Mode,
} from './types';
import {useQueryRunner} from './hooks/useQueryRunner';
import AppHeader from './components/AppHeader';
import DatabaseRail from './components/DatabaseRail';
import DatabaseDrawer from './components/DatabaseDrawer';
import QueryComposer from './components/QueryComposer';
import ProcessingState from './components/ProcessingState';
import ResultWorkspace from './components/ResultWorkspace';


export default function App() {
  const [databases, setDatabases] = useState<DatabaseSummary[]>([]);
  const [selectedId, setSelectedId] = useState('');
  const [database, setDatabase] = useState<DatabaseDetail | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [question, setQuestion] = useState('');
  const [evidence, setEvidence] = useState('');
  const [showEvidence, setShowEvidence] = useState(false);
  const [mode, setMode] = useState<Mode>('fast');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const query = useQueryRunner();

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    getDatabases()
      .then((items) => {
        setDatabases(items);
        const initial = items.find((item) => item.example_queries.length) ?? items[0];
        if (initial) setSelectedId(initial.db_id);
      })
      .catch(() => setDatabases([]));
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    getDatabase(selectedId).then(setDatabase).catch(() => setDatabase(null));
  }, [selectedId]);

  const selectedSummary = useMemo(
    () => databases.find((item) => item.db_id === selectedId),
    [databases, selectedId],
  );
  const examples = selectedSummary?.example_queries ?? [];

  const selectDatabase = (dbId: string) => {
    setSelectedId(dbId);
    setQuestion('');
    setEvidence('');
  };
  const useExample = (nextQuestion: string, nextEvidence: string) => {
    setQuestion(nextQuestion);
    setEvidence(nextEvidence);
    setShowEvidence(Boolean(nextEvidence));
  };

  return (
    <div className="app">
      <AppHeader health={health} />
      <div className="layout">
        <DatabaseRail
          dbs={databases}
          selected={selectedId}
          onSelect={(item) => selectDatabase(item.db_id)}
          onSchema={() => setDrawerOpen(true)}
          onMobile={() => setDrawerOpen(true)}
          onExample={useExample}
        />
        <main>
          <div className="workspace-head">
            <div>
              <p className="eyebrow">TEXT-TO-SQL WORKSPACE</p>
              <h2>Ask your data a question</h2>
              <p className="muted">
                Grounded generation with execution-verified answers.
              </p>
            </div>
            <span className="db-badge">
              <Database size={14} />
              {selectedId || 'Select a database'}
            </span>
          </div>
          <QueryComposer
            question={question}
            setQuestion={setQuestion}
            evidence={evidence}
            setEvidence={setEvidence}
            showEvidence={showEvidence}
            setShowEvidence={setShowEvidence}
            mode={mode}
            setMode={setMode}
            onRun={() => query.execute(selectedId, question, evidence, mode)}
            loading={query.loading}
            disabled={!selectedId || !question.trim()}
          />
          {examples.length > 0 && (
            <div className="examples">
              <span>Try an example</span>
              {examples.map((example) => (
                <button
                  key={example.question}
                  onClick={() => useExample(example.question, example.evidence)}
                >
                  {example.question}
                </button>
              ))}
            </div>
          )}
          {query.loading && <ProcessingState mode={mode} />}
          {query.error && (
            <div className="error">
              <Zap size={16} />
              {query.error}
            </div>
          )}
          {query.response && (
            <ResultWorkspace
              key={query.response.request_id}
              response={query.response}
              mode={query.response.mode}
            />
          )}
        </main>
      </div>
      {drawerOpen && (
        <DatabaseDrawer
          db={database}
          dbs={databases}
          onSelect={(dbId) => {
            selectDatabase(dbId);
            setDrawerOpen(false);
          }}
          onClose={() => setDrawerOpen(false)}
        />
      )}
    </div>
  );
}
