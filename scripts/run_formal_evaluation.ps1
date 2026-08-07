$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$queryPath = Join-Path $projectRoot "data\evaluation\public_dataset_200_formal_queries.csv"
$outputDir = Join-Path $projectRoot "outputs\evaluation_public_200_formal"

& (Join-Path $projectRoot ".venv\Scripts\python.exe") `
    (Join-Path $projectRoot "scripts\build_formal_query_set.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Write-Host "[1/3] BGE-M3 OCR text retrieval"
& (Join-Path $projectRoot ".venv\Scripts\python.exe") `
    (Join-Path $projectRoot "scripts\evaluate_text_retrieval.py") `
    --queries $queryPath `
    --manifest (Join-Path $projectRoot "outputs\user_library\manifest.jsonl") `
    --index-dir (Join-Path $projectRoot "outputs\user_library\text_index") `
    --model (Join-Path $projectRoot "models\bge-m3") `
    --output (Join-Path $outputDir "text_retrieval.json") `
    --device cuda:0 `
    --batch-size 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/3] Qwen3-VL visual retrieval"
& (Join-Path $projectRoot ".venv-vl\Scripts\python.exe") `
    (Join-Path $projectRoot "scripts\evaluate_visual_retrieval.py") `
    --queries $queryPath `
    --index-dir (Join-Path $projectRoot "outputs\user_library\visual_index") `
    --model (Join-Path $projectRoot "models\qwen3-vl-embedding-2b") `
    --output (Join-Path $outputDir "visual_retrieval.json") `
    --batch-size 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/3] Fixed and OCR-quality-adaptive fusion"
& (Join-Path $projectRoot ".venv\Scripts\python.exe") `
    (Join-Path $projectRoot "scripts\evaluate_fusion.py") `
    --queries $queryPath `
    --ocr-summary (Join-Path $projectRoot "outputs\user_library\ocr\summary.csv") `
    --text-scores (Join-Path $outputDir "text_score_matrix.npz") `
    --visual-scores (Join-Path $outputDir "visual_score_matrix.npz") `
    --output (Join-Path $outputDir "fusion_retrieval.json") `
    --fixed-text-weight 0.5 `
    --adaptive-text-weight-cap 0.6
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Formal evaluation completed: $outputDir"
