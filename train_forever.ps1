# Train in the port and keep going: if training ever stops with an error, resume it from the
# latest checkpoint after a short pause. Logs go to runs/logs/port.txt (appended) and port.err.
#   powershell -ExecutionPolicy Bypass -File train_forever.ps1 -Fresh     # start a new run
#   powershell -ExecutionPolicy Bypass -File train_forever.ps1            # continue runs/port
param([switch]$Fresh, [int]$Pop = 500, [int]$Workers = 20)
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
# cmd does the redirection so the log stays plain text (Windows PowerShell 5.1 would write UTF-16)
if ($Fresh) {
    cmd /c "$py -u train_port.py --fresh --seed-from runs/real --out runs/port --pop $Pop --workers $Workers >> runs\logs\port.txt 2>> runs\logs\port.err"
    $code = $LASTEXITCODE
} else {
    $code = 1
}
while ($code -ne 0) {
    cmd /c "echo --- training stopped (exit $code) at %date% %time%; resuming --- >> runs\logs\port.txt"
    Start-Sleep 15
    cmd /c "$py -u train_port.py --resume --out runs/port --pop $Pop --workers $Workers >> runs\logs\port.txt 2>> runs\logs\port.err"
    $code = $LASTEXITCODE
}
