#requires -Version 7.0
<#
Run-Verification.ps1
- Opens a new terminal (Windows Terminal if available; otherwise pwsh)
- CD to project directory
- Activates venv
- Runs python command redirecting all output to logs\verification.log
- Shows last lines of log in the terminal
- Keeps the terminal open until Enter is pressed
#>

$ErrorActionPreference = 'Stop'

$ProjectDir = 'E:\Workspaces\WS_Python\Portfoilo\portfolioProcessor'
$Activate   = Join-Path $ProjectDir '.venv\Scripts\Activate.ps1'
$LogsDir    = Join-Path $ProjectDir 'logs'
$OutDir     = Join-Path $ProjectDir 'out'
$LogFile    = Join-Path $LogsDir 'verification.log'

# Ensure folders exist
New-Item -ItemType Directory -Force -Path $LogsDir, $OutDir | Out-Null

# Write an inner script to a temp file (this avoids parser issues with -Command)
$TempScript = Join-Path $env:TEMP 'run_portfolio_verification.ps1'

$Inner = @"
Set-Location -LiteralPath '$ProjectDir'

Write-Host "Working directory: `$(Get-Location)" -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath '$Activate')) {
  Write-Error "Virtual environment activate script not found: $Activate"
  exit 1
}

. '$Activate'
Write-Host "Virtual environment activated." -ForegroundColor Green

Write-Host "Running: python .\caller.py all --show-audit --debug --outdir out > .\logs\verification.log 2>&1" -ForegroundColor Yellow
python .\caller.py all --show-audit --debug --outdir out > .\logs\verification.log 2>&1

`$code = `$LASTEXITCODE
Write-Host ""
Write-Host "Exit code: `$code" -ForegroundColor Cyan
Write-Host "Log file: $LogFile" -ForegroundColor Cyan

if (Test-Path -LiteralPath '$LogFile') {
  Write-Host ""
  Write-Host "=== Last 60 lines of verification.log ===" -ForegroundColor Magenta
  Get-Content -LiteralPath '$LogFile' -Tail 60
}

Write-Host ""
Write-Host "Press Enter to close..." -ForegroundColor DarkGray
[void](Read-Host)
exit `$code
"@

# PS7+ safe encoding (UTF-8 without BOM)
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
Set-Content -Path $TempScript -Value $Inner -Encoding $utf8NoBom

# Launch in a new terminal window
if (Get-Command wt.exe -ErrorAction SilentlyContinue) {
  # Windows Terminal
  wt pwsh -NoExit -File $TempScript
}
else {
  # Fallback to a new PowerShell 7 console window
  Start-Process -FilePath 'pwsh.exe' -ArgumentList @('-NoExit', '-File', $TempScript)
}
