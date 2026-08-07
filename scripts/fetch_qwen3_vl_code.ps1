$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$thirdPartyRoot = Join-Path $projectRoot "third_party"
$repoDir = Join-Path $thirdPartyRoot "Qwen3-VL-Embedding"
$repoUrl = "https://github.com/QwenLM/Qwen3-VL-Embedding.git"
$pinnedCommit = "393e2978d27852b0d0230d6994f37f9c15bed73c"

New-Item -ItemType Directory -Path $thirdPartyRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $repoDir ".git"))) {
    git clone --filter=blob:none --no-checkout $repoUrl $repoDir
}

git -C $repoDir fetch --depth 1 origin $pinnedCommit
git -C $repoDir checkout --detach $pinnedCommit

$actualCommit = git -C $repoDir rev-parse HEAD
if ($actualCommit -ne $pinnedCommit) {
    throw "Qwen3-VL-Embedding commit mismatch: $actualCommit"
}

Write-Output "Qwen3-VL-Embedding code ready at commit $actualCommit"
