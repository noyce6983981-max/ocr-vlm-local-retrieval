$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$vlPython = Join-Path $projectRoot ".venv-vl\Scripts\python.exe"
$requirements = Join-Path $projectRoot "requirements-vl.txt"

if (-not (Test-Path -LiteralPath $vlPython)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & $pyLauncher.Source -3.11 -m venv (Join-Path $projectRoot ".venv-vl")
    }
    else {
        $python = (Get-Command python -ErrorAction Stop).Source
        & $python -m venv (Join-Path $projectRoot ".venv-vl")
    }
}

& $vlPython -m pip install --no-cache-dir --upgrade pip `
    -i https://pypi.tuna.tsinghua.edu.cn/simple

& $vlPython -m pip install --no-cache-dir `
    "torch==2.8.0+cu128" `
    "torchvision==0.23.0+cu128" `
    -i https://pypi.tuna.tsinghua.edu.cn/simple `
    -f https://mirrors.aliyun.com/pytorch-wheels/cu128/

& $vlPython -m pip install --no-cache-dir -r $requirements `
    -i https://pypi.tuna.tsinghua.edu.cn/simple

& $vlPython -c "import torch, transformers, qwen_vl_utils, modelscope; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('transformers:', transformers.__version__)"
