param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$')]
    [string]$RunName
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-Project.ps1')
$experimentRoot = Join-Path $env:PROJECT_ROOT "artifacts\reproductions\$RunName"
if (Test-Path -LiteralPath $experimentRoot) {
    throw "Output already exists; choose a new RunName to preserve prior results: $experimentRoot"
}
$pythonExe = Join-Path $env:PROJECT_ROOT '.env\python.exe'
$realData = Join-Path $env:PROJECT_ROOT 'data\processed\pilot_pretraining_v1.npz'
$simData = Join-Path $env:PROJECT_ROOT 'data\processed\sim_adaptation_v1.npz'
foreach ($inputPath in @($pythonExe,$realData,$simData)) {
    if (-not (Test-Path -LiteralPath $inputPath)) { throw "Required local input missing: $inputPath" }
}
New-Item -ItemType Directory -Path $experimentRoot | Out-Null
$pretrainPath = Join-Path $experimentRoot 'visual_pretraining_v1'
$statePath = Join-Path $experimentRoot 'visual_state_v1'
$graspPath = Join-Path $experimentRoot 'visual_grasp_v1'
& $pythonExe scripts\run_tests.py
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& $pythonExe scripts\train_visual_ablation.py --data $realData --seeds 7 17 27 --output $pretrainPath
if ($LASTEXITCODE -ne 0) { throw 'Pretraining failed' }
& $pythonExe scripts\adapt_visual_state.py --data $simData --seeds 7 17 27 --pretraining $pretrainPath --output $statePath
if ($LASTEXITCODE -ne 0) { throw 'Adaptation failed' }
& $pythonExe scripts\evaluate_visual_grasp.py --seeds 7 17 27 --models $statePath --output $graspPath --record-first
if ($LASTEXITCODE -ne 0) { throw 'Physical evaluation failed' }
& $pythonExe scripts\analyze_visual_ablation.py --runs $experimentRoot --output (Join-Path $experimentRoot 'report')
if ($LASTEXITCODE -ne 0) { throw 'Analysis or provenance audit failed' }
Write-Output "Finished. Report: $(Join-Path $experimentRoot 'report\REPORT_ZH.md')"
