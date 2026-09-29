import type { Benchmark, Job } from './types'

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => fetch('/api/health').then(json<Record<string, unknown>>),
  benchmark: () => fetch('/api/benchmarks/summary').then(json<Benchmark>),
  job: (id: string) => fetch(`/api/extractions/${id}`).then(json<Job>),
  create: (file: File, bank: string, year: number) => {
    const form = new FormData()
    form.append('file', file)
    form.append('bank', bank)
    form.append('year', String(year))
    return fetch('/api/extractions', { method: 'POST', body: form }).then(json<{ job_id: string; status: string }>)
  },
}
