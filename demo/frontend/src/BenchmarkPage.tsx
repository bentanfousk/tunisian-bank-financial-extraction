import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { ArrowRight, BookOpen, ChevronDown, Cloud, HardDrive, LoaderCircle, LockKeyhole } from 'lucide-react'
import { api } from './api'
import type { Benchmark, Headline, Model } from './types'

const BenchmarkChart = lazy(() => import('./BenchmarkChart'))
const percent = (value: number | null | undefined) => value == null ? 'N/A' : `${(value * 100).toFixed(1)}%`
const number = (value: number | null | undefined, digits = 0) => value == null ? 'N/A' : value.toLocaleString('en-US', { maximumFractionDigits: digits })
const seconds = (value: number | null | undefined) => value == null ? 'N/A' : `${value.toFixed(1)} s`
const modelName = (model: Model) => model.id === 'ministral3_3b' ? 'Ministral 3B' : model.id === 'gemini-3.6-flash' ? 'Gemini Flash' : model.id.replaceAll('_', ' ')

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>
}

function OperationRow({ label, value }: { label: string; value: string }) {
  return <div className="operation-row"><span>{label}</span><strong>{value}</strong></div>
}

function ComparisonCard({ item, focus }: { item: Headline; focus: boolean }) {
  return <div className={`headline-card ${focus ? 'focus' : ''}`}>
    <div className="condition"><span>{item.system.includes('Gemini') ? 'CLOUD BASELINE' : 'LOCAL SYSTEM'}</span>{item.system.includes('Gemini') ? <Cloud size={16}/> : <HardDrive size={16}/>}</div>
    <h3>{item.system}</h3>
    <p>{item.input}</p>
    <div className="headline-score">{percent(item.accuracy)}</div>
    <div className="bar-track"><div style={{ width: percent(item.accuracy) }}/></div>
    <div className="card-foot"><strong>{item.correct}/{item.total} fields</strong><span>Valid JSON {percent(item.valid_json_rate)}</span></div>
  </div>
}

export default function BenchmarkPage() {
  const [data, setData] = useState<Benchmark | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState('All')

  useEffect(() => { api.benchmark().then(setData).catch(cause => setError(cause instanceof Error ? cause.message : 'Benchmark unavailable.')) }, [])
  const filtered = useMemo(() => data?.reports.filter(row => filter === 'All' || row.system === filter) || [], [data, filter])

  if (error) return <main className="main"><div className="page-wrap"><div className="alert">{error}</div></div></main>
  if (!data) return <main className="main"><div className="page-wrap loading"><LoaderCircle className="spin"/> Loading frozen benchmark artifacts…</div></main>

  const models = [...data.models].sort((a, b) => b.accuracy - a.accuracy)
  const selectedLocal = models.find(model => model.selected_for_v2 && model.source === 'local')
  const controlled = data.headline.filter(item => item.input.includes('Manual'))
  const complete = data.headline.filter(item => !item.input.includes('Manual'))
  const localControlled = controlled.find(item => !item.system.includes('Gemini'))
  const localComplete = complete.find(item => !item.system.includes('Gemini'))
  const cloudControlled = controlled.find(item => item.system.includes('Gemini'))
  const cloudComplete = complete.find(item => item.system.includes('Gemini'))
  const errorCounts = [...data.errors.counts].sort((a, b) => b.count - a.count)

  return <main className="main"><div className="page-wrap benchmark-page">
    <header className="page-top benchmark-intro"><div><span className="eyebrow">BENCHMARK</span><h1>Can a small local model approach cloud quality?</h1><p>Seven indicators from complete annual reports, tested from controlled extraction through end-to-end processing.</p></div><span className="frozen-badge"><LockKeyhole size={15}/> Saved results</span></header>

    <section className="benchmark-section benchmark-primary">
      <div className="section-title benchmark-heading"><div><span className="eyebrow">01 · MODEL SELECTION</span><h2>Which local model should power the system?</h2><p>All models share the same pages, prompt, schema and scoring; Gemini Flash is the cloud baseline.</p></div></div>
      <div className="chart-card"><Suspense fallback={<div className="chart-loading">Loading model chart…</div>}><BenchmarkChart models={models}/></Suspense>
        <div className="model-table"><div className="model-row heading"><span>Model</span><span>Correct</span><span>Valid JSON</span><span>Field citations</span><span>Latency</span><span>Tokens/s</span><span>Peak VRAM</span></div>{models.map(model => <div className="model-row" key={model.id}><span><strong>{modelName(model)}</strong>{model.selected_for_v2 && <small>Selected local model for V2</small>}</span><span>{model.correct}/{model.total}</span><span>{percent(model.valid_json_rate)}</span><span>{percent(model.citation_accuracy)}</span><span>{seconds(model.latency_seconds)}</span><span>{number(model.tokens_per_second, 1)}</span><span>{model.peak_vram_mb == null ? 'N/A' : `${number(model.peak_vram_mb)} MB`}</span></div>)}</div>
      </div>
      {selectedLocal && <div className="benchmark-takeaway"><strong>{modelName(selectedLocal)}, the strongest local candidate, was selected for V2</strong><span>{selectedLocal.correct}/{selectedLocal.total} fields · {percent(selectedLocal.accuracy)} strict accuracy</span></div>}
    </section>

    <section className="benchmark-section benchmark-transition">
      <div className="section-title benchmark-heading"><div><span className="eyebrow">02 · FROM CONTROLLED TO END TO END</span><h2>What changed in V2?</h2></div></div>
      <div className="flow-comparison">
        <div className="flow-row"><span className="flow-label">V1 · CONTROLLED</span><div className="flow-steps"><span>Selected financial pages</span><ArrowRight size={15}/><span>Ministral 3B</span><ArrowRight size={15}/><span>7 fields</span></div></div>
        <div className="flow-row"><span className="flow-label">V2 · END TO END</span><div className="flow-steps"><span>Complete annual report</span><ArrowRight size={15}/><span>BGE-M3</span><ArrowRight size={15}/><span>FAISS</span><ArrowRight size={15}/><span>Ministral 3B</span><ArrowRight size={15}/><span>7 fields</span></div></div>
      </div>
      <p className="benchmark-note">V2 removes manual page selection and makes the system retrieve its own evidence.</p>
    </section>

    <section className="benchmark-section benchmark-primary">
      <div className="section-title benchmark-heading"><div><span className="eyebrow">03 · SYSTEM COMPARISON</span><h2>What changes with the complete report?</h2><p>Strict field accuracy across controlled and complete-report inputs.</p></div></div>
      <div className="comparison-groups">
        <div className="comparison-group"><div className="comparison-group-title"><strong>CONTROLLED</strong><span>Selected financial pages</span></div><div className="headline-grid">{controlled.map(item => <ComparisonCard key={`${item.system}-${item.input}`} item={item} focus={false}/>)}</div></div>
        <div className="comparison-group"><div className="comparison-group-title"><strong>COMPLETE REPORT</strong><span>Full document input</span></div><div className="headline-grid">{complete.map(item => <ComparisonCard key={`${item.system}-${item.input}`} item={item} focus={!item.system.includes('Gemini')}/>)}</div></div>
      </div>
      {localControlled && localComplete && <div className="benchmark-takeaway comparison-takeaway"><strong>Local extraction: {percent(localControlled.accuracy)} → {percent(localComplete.accuracy)} after automatic retrieval.</strong>{cloudControlled && cloudComplete && <span>Cloud baseline: {percent(cloudControlled.accuracy)} → {percent(cloudComplete.accuracy)}.</span>}</div>}
    </section>

    <section className="benchmark-section benchmark-secondary">
      <div className="section-title benchmark-heading"><div><span className="eyebrow">04 · FAILURE ANALYSIS</span><h2>Where did the local system fail?</h2><p>{data.errors.rows.length} incorrect field predictions classified from the saved analysis.</p></div></div>
      <div className="analysis-panel"><div className="error-bars">{errorCounts.map(row => <div className="error-line" key={row.category}><div><strong>{row.category.replaceAll('_', ' ')}</strong><span>{row.count}</span></div><div className="bar-track"><div style={{ width: `${row.count / Math.max(...errorCounts.map(count => count.count)) * 100}%` }}/></div></div>)}</div><details className="error-details"><summary>Inspect field-level classifications <ChevronDown size={15}/></summary><div className="error-list">{data.errors.rows.map((row, index) => <div key={`${row.doc_id}-${row.field}-${index}`}><strong>{row.doc_id}</strong><span>{row.field.replaceAll('_', ' ')}</span><em>{row.category.replaceAll('_', ' ')}</em></div>)}</div></details></div>
    </section>

    <section className="benchmark-section benchmark-secondary">
      <div className="section-title benchmark-heading"><div><span className="eyebrow">05 · OPERATIONAL TRADE-OFF</span><h2>What is the operational trade-off?</h2><p>Local resources alongside the saved cloud usage and cost estimate.</p></div></div>
      <div className="operation-grid"><div className="operation-card"><div className="operation-title"><HardDrive size={18}/><strong>Local · RAG + Ministral</strong></div><OperationRow label="Mean end-to-end time" value={seconds(Number(data.local_operational.average_end_to_end_seconds))}/><OperationRow label="Maximum peak VRAM" value={`${number(Number(data.local_operational.maximum_gpu_peak_mb))} MB`}/><OperationRow label="Mean generation speed" value={`${number(Number(data.local_operational.average_tokens_per_second), 1)} tokens/s`}/><OperationRow label="Direct API charge" value="None"/></div><div className="operation-card"><div className="operation-title"><Cloud size={18}/><strong>Cloud · Gemini Flash</strong></div><OperationRow label="Mean end-to-end time" value={seconds(Number(data.cloud_operational.average_end_to_end_seconds))}/><OperationRow label="Token usage" value={`${number(Number(data.cloud_operational.total_input_tokens))} in / ${number(Number(data.cloud_operational.total_output_tokens))} out`}/><OperationRow label="Theoretical standard API cost" value={`$${Number(data.cloud_operational.total_standard_cost_usd).toFixed(3)}`}/></div></div>
      <p className="benchmark-note">Cloud cost is theoretical; hardware and electricity costs were not measured.</p>
    </section>

    <section className="benchmark-section benchmark-tertiary">
      <div className="section-title benchmark-heading"><div><span className="eyebrow">V2 · RETRIEVAL DETAIL</span><h2>What powered the retrieval step?</h2></div></div>
      <div className="research-metrics"><Metric label="Configuration" value={data.retrieval.configuration}/><Metric label="Embedding" value={data.retrieval.embedding_model}/><Metric label="Chunking" value={data.retrieval.chunking.replace('_', ' ')}/><Metric label="Pages per query" value={String(data.retrieval.pages_per_query)}/><Metric label="Authority page recall" value={percent(data.retrieval.authoritative_page_recall)}/><Metric label="Field evidence recall" value={percent(data.retrieval.field_evidence_recall)}/><Metric label="Contamination rate" value={percent(data.retrieval.contamination_rate)}/><Metric label="Mean retrieval time" value={seconds(Number(data.local_operational.average_retrieval_seconds ?? null))}/></div>
    </section>

    <section className="benchmark-section benchmark-tertiary">
      <div className="section-title benchmark-heading"><div><span className="eyebrow">REPORT-LEVEL EVIDENCE</span><h2>Detailed runs</h2></div><select aria-label="Filter system" value={filter} onChange={event => setFilter(event.target.value)}><option>All</option><option>RAG-4 + Ministral 3B</option><option>Gemini Flash</option></select></div>
      <div className="report-table-wrap"><table className="report-table"><thead><tr><th>Report</th><th>System</th><th>Correct</th><th>Strict accuracy</th><th>Valid JSON</th><th>Latency</th><th>Theoretical cost</th></tr></thead><tbody>{filtered.map(row => <tr key={`${row.report}-${row.system}`}><td><strong>{row.report}</strong></td><td>{row.system}</td><td>{row.correct == null ? 'N/A' : `${row.correct}/${row.total}`}</td><td>{percent(row.accuracy)}</td><td>{row.valid_json == null ? 'N/A' : row.valid_json ? 'Yes' : 'No'}</td><td>{seconds(row.latency_seconds)}</td><td>{row.theoretical_standard_cost_usd == null ? 'N/A' : `$${row.theoretical_standard_cost_usd.toFixed(3)}`}</td></tr>)}</tbody></table></div><p className="form-note">N/A means the selected final per-report artifact does not provide that value.</p>
    </section>
    <details className="collapsible sources"><summary><span><BookOpen size={18}/> Artifact sources <small>Files read by this dashboard</small></span><ChevronDown size={18}/></summary><ul>{data.sources.map(source => <li key={source}>{source}</li>)}</ul></details>
  </div></main>
}
