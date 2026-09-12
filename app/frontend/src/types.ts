export type Mode = 'fast' | 'accurate';

export type DemoQuery = {
  question: string;
  evidence: string;
};

export type DatabaseSummary = {
  db_id: string;
  split: 'train' | 'dev';
  table_count: number;
  column_count: number;
  example_queries: DemoQuery[];
};

export type SchemaColumn = {
  name: string;
  display_name: string;
  type: string;
};

export type SchemaTable = {
  name: string;
  display_name: string;
  columns: SchemaColumn[];
  primary_keys: string[];
};

export type ForeignKey = {
  source_table: string;
  source_column: string;
  target_table: string;
  target_column: string;
};

export type DatabaseDetail = {
  db_id: string;
  split: 'train' | 'dev';
  tables: SchemaTable[];
  foreign_keys: ForeignKey[];
  schema_text: string;
};

export type HealthResponse = {
  status: string;
  api_ready: boolean;
  model_ready: boolean;
  adapter_name: string;
  grounding_ready: boolean;
  database_count: number;
  version: string;
};

export type GroundingHit = {
  table: string;
  column: string;
  description: string;
  value_description: string;
  data_format: string;
  matched_values: unknown[];
  line: string;
};

export type Candidate = {
  index: number;
  kind: 'greedy' | 'sampled';
  reasoning: string;
  sql: string;
  raw_output: string;
  extraction_status: string;
  format_compliance: boolean;
  completion_tokens: number | null;
  mean_logprob: number | null;
  execution_status: string;
  row_count: number;
  column_count: number;
  empty_result: boolean;
  execution_elapsed_ms: number;
  error_type: string | null;
  selected: boolean;
  cluster_support: number;
};

export type ExecutionView = {
  status: string;
  columns: string[];
  rows: unknown[][] | null;
  row_count: number;
  displayed_row_count: number;
  result_truncated: boolean;
  elapsed_ms: number;
  error_type: string | null;
  error_message: string | null;
  accessed_tables: string[];
  accessed_columns: {table: string; column: string}[];
};

export type VoteInfo = {
  candidate_count: number;
  executable_count: number;
  chosen_index: number | null;
  support: number;
  cluster_count: number;
  used_vote: boolean;
  reason: string;
};

export type QueryTimings = {
  grounding_ms: number;
  prompt_ms: number;
  generation_ms: number;
  candidate_execution_ms: number;
  vote_ms: number;
  display_execution_ms: number;
  total_ms: number;
};

export type QueryResponse = {
  request_id: string;
  status: string;
  db_id: string;
  split: 'train' | 'dev';
  mode: Mode;
  question: string;
  grounding: GroundingHit[];
  candidates: Candidate[];
  selected_candidate: Candidate | null;
  execution: ExecutionView | null;
  vote: VoteInfo | null;
  timings: QueryTimings;
};
