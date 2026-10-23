param(
    [string]$EdgePath = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root

if (-not (Test-Path -LiteralPath $EdgePath)) {
    Pop-Location
    throw "Microsoft Edge was not found at '$EdgePath'. Pass -EdgePath with the correct path."
}

Write-Host "Building report HTML from results.json ..."
python scripts\build_report.py

$reportHtml = Join-Path $Root "reports\POS_Tagging_Report.html"
$reportPdf = Join-Path $Root "reports\POS_Tagging_Report.pdf"
$reportUri = ([System.Uri]$reportHtml).AbsoluteUri

Write-Host "Printing report to PDF ..."
& $EdgePath --headless=new --disable-gpu --no-pdf-header-footer "--print-to-pdf=$reportPdf" $reportUri | Out-Null
Start-Sleep -Seconds 3

Write-Host "Exporting executed notebook to HTML ..."
python -m jupyter nbconvert --to html "notebooks\POS_Tagging_Comparison.ipynb" --output-dir "reports" --output "output_screenshots"

$shotsHtml = Join-Path $Root "reports\output_screenshots.html"
$shotsPdf = Join-Path $Root "reports\output_screenshots.pdf"
$shotsUri = ([System.Uri]$shotsHtml).AbsoluteUri

Write-Host "Printing notebook outputs to PDF ..."
& $EdgePath --headless=new --disable-gpu --no-pdf-header-footer "--print-to-pdf=$shotsPdf" $shotsUri | Out-Null
Start-Sleep -Seconds 3

Pop-Location

Get-Item -LiteralPath $reportPdf, $shotsPdf | Select-Object Name, Length, LastWriteTime
