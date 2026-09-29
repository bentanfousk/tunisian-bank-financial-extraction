export type Field = {
  key: string; label: string; status: string; value: number | null;
  unit_multiplier: number | null; scope: string; source_year: number;
  pdf_page: number | null; report_page: number | null; evidence: string; notes?: string | null
}
export type RetrievedPage = {
  pdf_page: number; report_page: number | null; score: number;
  matches: { query_id: string; rank: number; score: number }[]; excerpt: string
}
export type ExtractionResult = {
  job_id: string;
  document: { filename: string; page_count: number; bank?: string; target_year: number };
  system: { model: string; embedding_model: string; rag_configuration: string; local: boolean };
  retrieval: { pages: RetrievedPage[] }; fields: Field[]; raw_prediction: unknown;
  runtime: Record<string, number | null>;
  validation: { first_pass_valid: boolean; final_valid: boolean; retry_used: boolean }
}
export type JobEvent = { status: string; message: string; timestamp: string; data: Record<string, unknown> }
export type Job = {
  job_id: string; status: string; filename: string; bank: string; year: number;
  result: ExtractionResult | null; error: string | null; events: JobEvent[]
}
export type Headline = { system: string; input: string; prompt: string; documents: number; correct: number; total: number; accuracy: number; valid_json_rate: number; citation_accuracy: number }
export type Model = { id: string; source: string; model_tag: string | null; correct: number; total: number; accuracy: number; valid_json_rate: number; first_pass_valid_rate: number | null; citation_accuracy: number | null; latency_seconds: number | null; tokens_per_second: number | null; peak_vram_mb: number | null; selected_for_v2: boolean }
export type ErrorRow = { doc_id: string; field: string; category: string; detail: string }
export type ReportRow = { report: string; system: string; input: string; correct: number | null; total: number; accuracy: number | null; valid_json: boolean | null; first_pass_valid: boolean | null; latency_seconds: number | null; theoretical_standard_cost_usd: number | null }
export type Benchmark = {
  headline: Headline[]; models: Model[];
  errors: { counts: { category: string; count: number }[]; rows: ErrorRow[] };
  reports: ReportRow[];
  retrieval: { configuration: string; embedding_model: string; chunking: string; pages_per_query: number; maximum_pages: number; candidate_k: number; authoritative_page_recall: number; field_evidence_recall: number; contamination_rate: number };
  local_operational: Record<string, number | string | null>;
  cloud_operational: Record<string, number | string | null>;
  sources: string[]
}
