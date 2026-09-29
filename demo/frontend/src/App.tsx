import { useEffect, useRef, useState } from 'react'
import { Activity, ArrowRight, BookOpen, Check, ChevronDown, CircleAlert, ClipboardCopy, Cpu, Database, FileSearch, FileText, LoaderCircle, LockKeyhole, Search, UploadCloud, X } from 'lucide-react'
import { api } from './api'
import BenchmarkPage from './BenchmarkPage'
import type { Field, Job, RetrievedPage } from './types'

const number = (value: number | null | undefined, digits = 0) => value == null ? 'N/A' : value.toLocaleString('en-US', { maximumFractionDigits: digits })
const seconds = (value: number | null | undefined) => value == null ? 'N/A' : `${value.toFixed(1)} s`
const unit = (multiplier: number | null) => multiplier == null ? 'N/A' : multiplier === 1 ? 'TND' : multiplier === 1000 ? 'Thousands of TND' : 'Millions of TND'
const stageLabel: Record<string, string> = {
  queued: 'Queued', loading_pdf: 'PDF loaded', extracting_pages: 'Pages extracted',
  embedding: 'Embeddings generated', indexing: 'FAISS index built', retrieving: 'Evidence retrieved',
  building_context: 'Context built', generating: 'Local extraction', retrying: 'Format repair',
  validating: 'JSON validation', completed: 'Complete', failed: 'Failed',
}
const stages = ['loading_pdf', 'extracting_pages', 'embedding', 'indexing', 'retrieving', 'building_context', 'generating', 'validating', 'completed']

function TopNav({ page, setPage }: { page: 'extract' | 'benchmark'; setPage: (p: 'extract' | 'benchmark') => void }) {
  return <aside className="sidebar">
    <div className="brand"><span className="brand-mark">fr<span>.</span></span><span><strong>Financial Report<br/>Intelligence</strong><small>Extraction suite</small></span></div>
    <nav aria-label="Primary navigation">
      <button className={page === 'extract' ? 'nav active' : 'nav'} onClick={() => setPage('extract')}><FileSearch size={18}/> Extraction <ArrowRight size={15} className="nav-arrow"/></button>
      <button className={page === 'benchmark' ? 'nav active' : 'nav'} onClick={() => setPage('benchmark')}><Activity size={18}/> Benchmark <ArrowRight size={15} className="nav-arrow"/></button>
    </nav>
    <div className="sidebar-foot"><div className="sidebar-foot-icon"><LockKeyhole size={16}/></div><div><strong>Local engine</strong><span>Extraction runs on this device.</span></div></div>
  </aside>
}

function StatusPill({ status }: { status: string }) {
  return <span className={`status-pill ${status === 'ready' ? 'ok' : status === 'needs_attention' ? 'warn' : ''}`}><span className="status-dot"/>{status === 'ready' ? 'Ready to process' : status === 'needs_attention' ? 'Setup needed' : 'Checking system'}</span>
}

function Evidence({ field, jobId, close }: { field: Field; jobId: string; close: () => void }) {
  return <div className="drawer-backdrop" onMouseDown={close}><section className="drawer" role="dialog" aria-modal="true" aria-label={`${field.label} evidence`} onMouseDown={e => e.stopPropagation()}>
    <div className="drawer-head"><div><span className="eyebrow">SOURCE EVIDENCE</span><h2>{field.label}</h2></div><button className="icon-button" onClick={close} aria-label="Close evidence"><X size={19}/></button></div>
    <div className="drawer-value">{field.value == null ? field.status.replace('_', ' ') : number(field.value, 12)}</div>
    <p className="drawer-unit">{unit(field.unit_multiplier)}</p>
    <div className="detail-grid"><div><span>Status</span><strong>{field.status.replace('_', ' ')}</strong></div><div><span>Scope</span><strong>{field.scope}</strong></div><div><span>Fiscal year</span><strong>{field.source_year}</strong></div><div><span>Report page</span><strong>{field.report_page ?? 'N/A'}</strong></div><div><span>PDF page</span><strong>{field.pdf_page ?? 'N/A'}</strong></div><div><span>Schema key</span><strong className="mono">{field.key}</strong></div></div>
    <div className="evidence-block"><span className="eyebrow">MODEL EVIDENCE · AS RETURNED</span><blockquote>{field.evidence || 'No evidence text returned.'}</blockquote><p className="muted">Open the cited PDF page to verify the exact source row and formatting.</p>{field.notes && <p className="muted">{field.notes}</p>}</div>
    {field.pdf_page && <><a className="secondary-button pdf-link" href={`/api/extractions/${jobId}/pdf#page=${field.pdf_page}`} target="_blank" rel="noreferrer"><FileText size={16}/> Open PDF page {field.pdf_page}</a><iframe className="pdf-frame" title={`PDF page ${field.pdf_page}`} src={`/api/extractions/${jobId}/pdf#page=${field.pdf_page}`}/></>}
    {!field.pdf_page && field.report_page != null && <p className="muted">The reported page could not be mapped uniquely to a physical PDF page.</p>}
  </section></div>
}

function ExtractionPage({ health }: { health: Record<string, unknown> | null }) {
  const [file, setFile] = useState<File | null>(null)
  const [bank, setBank] = useState('')
  const [year, setYear] = useState('')
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<Field | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const saved = sessionStorage.getItem('financial-demo-job')
    if (saved) api.job(saved).then(setJob).catch(() => sessionStorage.removeItem('financial-demo-job'))
  }, [])

  const choose = (picked?: File) => {
    if (!picked) return
    setError(null)
    if (!picked.name.toLowerCase().endsWith('.pdf')) { setError('Select a PDF annual report.'); return }
    setFile(picked)
    const match = picked.name.replace(/\.pdf$/i, '').match(/^([A-Za-z0-9_-]+)_(20\d\d)$/)
    if (match) { setBank(match[1]); setYear(match[2]) }
  }
  const submit = async () => {
    if (!file) return
    setError(null); setJob(null)
    try {
      const created = await api.create(file, bank.trim(), Number(year))
      sessionStorage.setItem('financial-demo-job', created.job_id)
      setJob(await api.job(created.job_id))
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Upload failed.') }
  }
  useEffect(() => {
    if (!job || ['completed', 'failed'].includes(job.status)) return
    const source = new EventSource(`/api/extractions/${job.job_id}/events`)
    const interval = window.setInterval(() => api.job(job.job_id).then(setJob).catch(() => {}), 2500)
    source.onmessage = () => api.job(job.job_id).then(next => {
      setJob(next)
      if (['completed', 'failed'].includes(next.status)) { source.close(); clearInterval(interval) }
    }).catch(() => {})
    source.onerror = () => source.close()
    return () => { source.close(); clearInterval(interval) }
  }, [job?.job_id])
  const result = job?.result
  const lastStage = job?.events.slice().reverse().find(event => stages.includes(event.status))?.status
  const currentIndex = stages.indexOf(job?.status === 'retrying' || job?.status === 'failed' ? (lastStage || '') : (job?.status || ''))

  return <main className="main"><div className="page-wrap">
    <div className="page-top"><div><span className="eyebrow">EXTRACTION</span><h1>Extract report data</h1><p>Upload an annual report and review seven indicators with source evidence.</p></div><StatusPill status={String(health?.status || 'checking')}/></div>
    <div className="tech-line"><span>PROCESSING STACK</span><span>BGE-M3 embeddings</span><span>FAISS retrieval</span><span>Ministral 3B extraction</span></div>
    <section className="upload-card">
      <div className="upload-heading"><div><span className="eyebrow">SOURCE FILE</span><h2>Annual report</h2></div><span>Text-based PDF · Up to 100 MB</span></div>
      <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0]) }}>
        <div className="upload-icon"><UploadCloud size={24}/></div><h2>{file ? file.name : 'Drop your PDF here'}</h2><p>{file ? `${number(file.size / 1024 / 1024, 1)} MB · Ready to process` : 'Or browse your files to get started'}</p>
        <input ref={inputRef} type="file" accept="application/pdf,.pdf" hidden onChange={e => choose(e.target.files?.[0])}/><button className="secondary-button" onClick={() => inputRef.current?.click()}>{file ? 'Change file' : 'Choose PDF'}</button>
      </div>
      <div className="metadata-row"><label>Bank identifier<input value={bank} onChange={e => setBank(e.target.value)} placeholder="e.g. UIB" aria-label="Bank identifier"/></label><label>Fiscal year<input value={year} onChange={e => setYear(e.target.value)} placeholder="2024" inputMode="numeric" aria-label="Fiscal year"/></label><button className="primary-button" disabled={!file || !/^[A-Za-z0-9_-]{2,40}$/.test(bank.trim()) || !/^20\d\d$/.test(year) || !!job && !['completed', 'failed'].includes(job.status)} onClick={submit}>Extract seven fields <ArrowRight size={16}/></button></div>
      <p className="form-note">Bank and year guide retrieval. A filename like UIB_2024.pdf fills them in automatically.</p>
      {error && <div className="alert"><CircleAlert size={17}/>{error}</div>}
    </section>
    {job && <section className="process-card"><div className="section-title"><div><span className="eyebrow">PROCESSING</span><h2>Pipeline activity</h2></div><span className="job-id">Job {job.job_id.slice(0, 8)}</span></div><div className="stage-grid">{stages.map((stage, index) => { const done = job.status === 'completed' || currentIndex > index; const active = job.status === stage || (job.status === 'retrying' && stage === 'validating'); return <div className={`stage ${done ? 'done' : ''} ${active ? 'current' : ''}`} key={stage}><span className="stage-icon">{done ? <Check size={15}/> : active ? <LoaderCircle size={15} className="spin"/> : <span className="small-dot"/>}</span><span>{stageLabel[stage]}</span></div> })}</div><div className="event-log">{job.events.slice(-4).map((event, index) => <div key={`${event.timestamp}-${index}`}><span>{new Date(event.timestamp).toLocaleTimeString()}</span><strong>{event.message}</strong>{Object.entries(event.data).map(([key, value]) => <em key={key}>{key.replaceAll('_', ' ')}: {String(value)}</em>)}</div>)}</div>{job.error && <div className="alert"><CircleAlert size={17}/>{job.error}</div>}</section>}
    {result && <><section className="results-section"><div className="section-title"><div><span className="eyebrow">STRUCTURED RESULT</span><h2>Seven financial indicators</h2><p>{result.document.filename} · {result.document.page_count} pages · Fiscal year {result.document.target_year}</p></div><span className="valid-badge"><Check size={15}/> Validated JSON</span></div><div className="result-table-wrap"><table className="result-table"><thead><tr><th>Financial indicator</th><th>Reported value</th><th>Unit</th><th>Year / Scope</th><th>Source</th><th>Status</th></tr></thead><tbody>{result.fields.map(field => <tr key={field.key} onClick={() => setSelected(field)} tabIndex={0} onKeyDown={e => { if (e.key === 'Enter') setSelected(field) }}><td><strong>{field.label}</strong><small>{field.key}</small></td><td className="value-cell">{field.value == null ? '—' : number(field.value, 12)}</td><td>{unit(field.unit_multiplier)}</td><td>{field.source_year}<small>{field.scope}</small></td><td>{field.report_page != null ? `Report p. ${field.report_page}` : field.pdf_page != null ? `PDF p. ${field.pdf_page}` : 'N/A'}<small>View evidence →</small></td><td><span className={`field-status ${field.status}`}>{field.status.replace('_', ' ')}</span></td></tr>)}</tbody></table></div></section>
      <section className="summary-card"><div className="section-title"><div><span className="eyebrow">EXECUTION</span><h2>Technical summary</h2></div></div><div className="stat-grid"><Metric label="Document" value={`${result.document.page_count} pages`}/><Metric label="Retrieved" value={`${result.retrieval.pages.length} pages`}/><Metric label="Embedding" value={result.system.embedding_model}/><Metric label="Generator" value={result.system.model}/><Metric label="Total time" value={seconds(result.runtime.total_seconds)}/><Metric label="Peak VRAM" value={result.runtime.peak_vram_mb == null ? 'N/A' : `${number(result.runtime.peak_vram_mb)} MB`}/></div><p className="form-note">Execution is local. Runtime figures are measured for this upload.</p></section>
      <details className="collapsible"><summary><span><Search size={18}/> Retrieved pages <small>{result.retrieval.pages.length} selected by semantic retrieval</small></span><ChevronDown size={18}/></summary><div className="retrieved-grid">{result.retrieval.pages.map((page: RetrievedPage) => <div className="retrieved-card" key={page.pdf_page}><div><strong>PDF page {page.pdf_page}</strong><span>{page.report_page == null ? 'Report page N/A' : `Report page ${page.report_page}`}</span></div><p>Queries: {page.matches.map(m => m.query_id.replace('_', ' ')).join(', ')}</p><p>Cosine score: {page.score.toFixed(3)}</p><blockquote>{page.excerpt}</blockquote></div>)}</div></details>
      <details className="collapsible"><summary><span><BookOpen size={18}/> View structured JSON <small>Validated original prediction</small></span><ChevronDown size={18}/></summary><div className="json-top"><button className="secondary-button" onClick={() => navigator.clipboard.writeText(JSON.stringify(result.raw_prediction, null, 2))}><ClipboardCopy size={15}/> Copy JSON</button></div><pre>{JSON.stringify(result.raw_prediction, null, 2)}</pre></details>
    </>}
    {!job && <section className="workflow"><div className="workflow-heading"><h2>How extraction works</h2><span>From source document to validated fields</span></div><div className="explainer"><div><FileText size={19}/><strong>Read the report</strong><span>Complete PDF</span></div><div><Cpu size={19}/><strong>Map its pages</strong><span>BGE-M3 embeddings</span></div><div><Database size={19}/><strong>Find evidence</strong><span>FAISS retrieval</span></div><div><FileSearch size={19}/><strong>Extract the fields</strong><span>Ministral 3B</span></div></div></section>}
  </div>{selected && job && <Evidence field={selected} jobId={job.job_id} close={() => setSelected(null)}/>}</main>
}

function Metric({ label, value }: { label: string; value: string }) { return <div className="metric"><span>{label}</span><strong>{value}</strong></div> }

export default function App() {
  const [page, setPage] = useState<'extract' | 'benchmark'>('extract')
  const [health, setHealth] = useState<Record<string, unknown> | null>(null)
  useEffect(() => { api.health().then(setHealth).catch(() => setHealth({ status: 'needs_attention', issues: ['Backend unavailable.'] })) }, [])
  return <div className="app"><TopNav page={page} setPage={setPage}/>{page === 'extract' ? <ExtractionPage health={health}/> : <BenchmarkPage/>}</div>
}
