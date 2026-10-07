$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PROJECT_ROOT = $projectRoot
$env:TEMP = Join-Path $projectRoot 'work\tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
$env:CONDA_PKGS_DIRS = Join-Path $projectRoot '.cache\conda\pkgs'
$env:CONDA_ENVS_PATH = Join-Path $projectRoot '.cache\conda\envs'
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache\matplotlib'
$env:NUMBA_CACHE_DIR = Join-Path $projectRoot '.cache\numba'
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CONDA_PREFIX = Join-Path $projectRoot '.env'
$projectBins = @($env:CONDA_PREFIX, (Join-Path $env:CONDA_PREFIX 'Scripts'), (Join-Path $env:CONDA_PREFIX 'Library\bin'))
$env:PATH = ($projectBins -join ';') + ';' + $env:PATH
foreach ($folder in @($env:TEMP,$env:PIP_CACHE_DIR,$env:HF_HOME,$env:TORCH_HOME,$env:MPLCONFIGDIR,$env:NUMBA_CACHE_DIR)) {
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
}
Set-Location -LiteralPath $projectRoot
