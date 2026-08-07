$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $projectRoot "correction_app.py"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "未找到项目虚拟环境：$pythonPath"
}

Set-Location -LiteralPath $projectRoot
& $pythonPath -m streamlit run $appPath `
    --server.address 127.0.0.1 `
    --server.port 8502 `
    --server.headless true `
    --browser.gatherUsageStats false
