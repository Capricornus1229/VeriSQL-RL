import type {
  DatabaseDetail,
  DatabaseSummary,
  HealthResponse,
  Mode,
  QueryResponse,
} from '../types';

export type QueryPayload = {
  db_id: string;
  question: string;
  evidence?: string;
  mode: Mode;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: {'Content-Type': 'application/json'},
    ...init,
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail ?? message;
    } catch {
      // Keep the HTTP status fallback when the body is not JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const getHealth = () => request<HealthResponse>('/health');
export const getDatabases = () => request<DatabaseSummary[]>('/databases');
export const getDatabase = (id: string) =>
  request<DatabaseDetail>(`/databases/${encodeURIComponent(id)}`);
export const runQuery = (payload: QueryPayload) =>
  request<QueryResponse>('/query', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
