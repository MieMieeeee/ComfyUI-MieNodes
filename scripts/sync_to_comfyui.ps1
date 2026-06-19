# Sync ComfyUI-MieNodes -> live ComfyUI custom_nodes
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot
python "$ScriptDir\sync_to_comfyui.py" @args
exit $LASTEXITCODE
