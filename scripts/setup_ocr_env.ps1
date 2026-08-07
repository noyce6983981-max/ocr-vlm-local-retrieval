$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$ocrPython = Join-Path $projectRoot ".venv-paddle\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $ocrPython)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & $pyLauncher.Source -3.11 -m venv (Join-Path $projectRoot ".venv-paddle")
    }
    else {
        $python = (Get-Command python -ErrorAction Stop).Source
        & $python -m venv (Join-Path $projectRoot ".venv-paddle")
    }
}

& $ocrPython -m pip install --no-cache-dir --upgrade pip `
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# Pin NumPy before both frameworks so pip does not need to replace it later.
& $ocrPython -m pip install --no-cache-dir "numpy==2.3.5" `
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ModelScope imports torch while PaddleOCR starts. Use CPU-only torch in the
# OCR environment so it cannot collide with PaddlePaddle's CUDA libraries.
& $ocrPython -m pip install --no-cache-dir "torch==2.5.1+cpu" `
    -i https://pypi.tuna.tsinghua.edu.cn/simple `
    -f https://mirrors.aliyun.com/pytorch-wheels/cpu/

& $ocrPython -m pip install --no-cache-dir "paddlepaddle-gpu==3.2.2" `
    -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

& $ocrPython -m pip install --no-cache-dir "paddleocr==3.7.0" `
    -i https://pypi.tuna.tsinghua.edu.cn/simple

& $ocrPython -c "import torch, paddle, paddleocr; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); print('paddle:', paddle.__version__, paddle.device.get_device()); print('paddleocr:', paddleocr.__version__)"
