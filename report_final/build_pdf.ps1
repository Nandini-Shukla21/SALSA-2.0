# Render SALSA_2.0_DRDO_SAG_Final_Report.html to a print-ready A4 PDF.
#
# Uses headless Chromium (Google Chrome, falling back to Microsoft Edge) to
# render the real HTML, with Paged.js performing CSS Paged Media pagination so
# that the running page numbers and the page numbers inside the Table of
# Contents, List of Figures and List of Tables are generated from the actual
# layout rather than written by hand.
#
# Run:  powershell -ExecutionPolicy Bypass -File report_final\build_pdf.ps1

$ErrorActionPreference = "Stop"

$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$html = Join-Path $dir "SALSA_2.0_DRDO_SAG_Final_Report.html"
$pdf  = Join-Path $dir "SALSA_2.0_DRDO_SAG_Final_Report.pdf"

if (-not (Test-Path $html)) { throw "Source not found: $html" }

$browsers = @(
  "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
  "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
  "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe",
  "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
  "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe"
)
$browser = $null
foreach ($b in $browsers) { if (Test-Path $b) { $browser = $b; break } }
if ($null -eq $browser) { throw "No Chromium-based browser found for PDF rendering." }
Write-Output "Renderer: $browser"

# Chromium's command line splits on spaces, and this project lives under a path
# that contains them ("SALSA 2.0"). Stage the document in a space-free working
# directory, render there, and copy the result back.
$work = Join-Path $env:TEMP "salsa2pdf"
if (Test-Path $work) { Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force $work | Out-Null
New-Item -ItemType Directory -Force (Join-Path $work "assets") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $work "SALSA_2.0_Report_Figures") | Out-Null

Copy-Item $html (Join-Path $work "report.html") -Force
Copy-Item (Join-Path $dir "assets\*")                   (Join-Path $work "assets") -Force
Copy-Item (Join-Path $dir "SALSA_2.0_Report_Figures\*") (Join-Path $work "SALSA_2.0_Report_Figures") -Force

$stagedHtml = Join-Path $work "report.html"
$stagedPdf  = Join-Path $work "report.pdf"
$uri = ([System.Uri]$stagedHtml).AbsoluteUri
Write-Output "Source:   $uri"

if (Test-Path $pdf) { Remove-Item $pdf -Force }

$profileDir = Join-Path $work "chromeprofile"

$args = @(
  "--headless=new",
  "--disable-gpu",
  "--no-sandbox",
  "--no-first-run",
  "--no-default-browser-check",
  "--disable-extensions",
  "--user-data-dir=$profileDir",
  "--allow-file-access-from-files",
  "--run-all-compositor-stages-before-draw",
  "--virtual-time-budget=120000",     # let Paged.js finish laying out every page
  "--no-pdf-header-footer",
  "--print-to-pdf=$stagedPdf",
  $uri
)

$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $browser -ArgumentList $args -NoNewWindow -PassThru -Wait
Write-Output "Renderer exited with code $($p.ExitCode) after $([math]::Round($sw.Elapsed.TotalSeconds,1))s"

if (-not (Test-Path $stagedPdf)) { throw "PDF was not produced." }
Copy-Item $stagedPdf $pdf -Force
$kb = [math]::Round((Get-Item $pdf).Length / 1KB, 1)
Write-Output "Wrote $pdf  ($kb KB)"
Write-Output "PDF_OK"
