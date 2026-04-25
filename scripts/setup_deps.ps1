param(
    [switch]$TrainingDeps,
    [switch]$ForceInstall
)

$ErrorActionPreference = "Stop"

$runtimePython = "C:\Users\bokde\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$python = if (Test-Path -LiteralPath $runtimePython) { $runtimePython } else { "python" }

function Install-Requirements {
    param(
        [string]$FilePath
    )

    Write-Host "[deps] pip install -r $FilePath"
    & $python -m pip install -r $FilePath
    if ($LASTEXITCODE -eq 0) {
        return
    }

    Write-Host "[deps] global install failed; retrying with --user for $FilePath"
    & $python -m pip install --user -r $FilePath
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed for $FilePath"
    }
}

function Test-Import {
    param(
        [string]$ModulesCsv
    )

    & $python -c "import $ModulesCsv"
    return $LASTEXITCODE -eq 0
}

Write-Host "[deps] python: $python"
& $python --version

if (-not $ForceInstall -and (Test-Import -ModulesCsv "openenv, fastapi, uvicorn, requests, pydantic, yaml, openai")) {
    Write-Host "[deps] runtime imports already available; skipping install."
}
else {
    Write-Host "[deps] installing runtime requirements..."
    Install-Requirements -FilePath "requirements.txt"
}

if ($TrainingDeps) {
    if (-not $ForceInstall -and (Test-Import -ModulesCsv "trl, transformers, datasets, matplotlib, pandas")) {
        Write-Host "[deps] training imports already available; skipping install."
    }
    else {
        Write-Host "[deps] installing training requirements..."
        Install-Requirements -FilePath "requirements-train.txt"
    }
}

Write-Host "[deps] validating imports..."
& $python -c "import openenv, fastapi, uvicorn, requests, pydantic, yaml, openai; print('runtime imports ok')"

if ($TrainingDeps) {
    Write-Host "[deps] validating training imports..."
    & $python -c "import trl, transformers, datasets, matplotlib, pandas; print('training imports ok')"
}

Write-Host "[deps] done."
