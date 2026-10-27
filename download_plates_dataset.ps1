# Downloads the Kaggle dataset using the Python downloader shipped with this project.
$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ScriptDir '.venv\Scripts\python.exe'

if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = 'python'
}

& $Python (Join-Path $ScriptDir 'download_plates_dataset.py') @args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
