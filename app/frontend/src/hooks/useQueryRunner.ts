import {useState} from 'react';

import {runQuery} from '../api/client';
import type {Mode, QueryResponse} from '../types';


export function useQueryRunner() {
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const execute = async (
    dbId: string,
    question: string,
    evidence: string,
    mode: Mode,
  ) => {
    setLoading(true);
    setError('');
    setResponse(null);
    try {
      setResponse(await runQuery({db_id: dbId, question, evidence, mode}));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to run query');
    } finally {
      setLoading(false);
    }
  };
  return {response, loading, error, execute};
}
