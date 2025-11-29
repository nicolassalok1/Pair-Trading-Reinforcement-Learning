Param(
    [string]$PythonExe = "python",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$testScript = Join-Path $PSScriptRoot "test-app.ps1"
if (-not (Test-Path $testScript)) {
    throw "Missing test script: $testScript"
}

& $testScript -PythonExe $PythonExe -SkipInstall:$SkipInstall
