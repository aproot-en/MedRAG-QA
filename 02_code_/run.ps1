# Convenience wrapper to run the GraphRAG + BGE-M3 medical QA experiment.
# Usage: .\run.ps1 [extra args passed to run_experiment.py]
#   .\run.ps1
#   .\run.ps1 --max-docs 2000 --num-eval 50 --rebuild
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
python run_experiment.py @args
