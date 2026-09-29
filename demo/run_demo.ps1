$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$frontend = Join-Path $PSScriptRoot 'frontend'
$runtime = Join-Path $repo 'demo_runtime'

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found at $python. See demo/README.md." }
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'npm was not found. Install Node.js and rerun.' }
if (-not (Test-Path -LiteralPath (Join-Path $frontend 'node_modules'))) { throw 'Frontend packages are missing. Run npm install in demo/frontend.' }
New-Item -ItemType Directory -Path $runtime -Force | Out-Null

try {
    $ollama = Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 -UseBasicParsing
    Write-Host 'Ollama is reachable.'
} catch {
    Write-Warning 'Ollama is not reachable. The benchmark will work; live extraction needs Ollama.'
}

$backend = Start-Process -FilePath $python -ArgumentList '-m', 'uvicorn', 'demo.backend.app.main:app', '--host', '127.0.0.1', '--port', '8000' -WorkingDirectory $repo -RedirectStandardOutput (Join-Path $runtime 'backend.stdout.log') -RedirectStandardError (Join-Path $runtime 'backend.stderr.log') -WindowStyle Hidden -PassThru
$web = Start-Process -FilePath 'npm.cmd' -ArgumentList 'run', 'dev' -WorkingDirectory $frontend -RedirectStandardOutput (Join-Path $runtime 'frontend.stdout.log') -RedirectStandardError (Join-Path $runtime 'frontend.stderr.log') -WindowStyle Hidden -PassThru

Write-Host "Backend PID: $($backend.Id)"
Write-Host "Frontend PID: $($web.Id)"
Write-Host 'Demo URL: http://127.0.0.1:5173'
Write-Host "Logs: $runtime"
Write-Host 'Open the URL in your browser.'
try {
    Read-Host 'Press Enter to stop both demo servers' | Out-Null
} finally {
    # npm.cmd launches Node as a child process; stop the whole launcher tree.
    taskkill.exe /T /F /PID $web.Id 2>$null | Out-Null
    Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
}
