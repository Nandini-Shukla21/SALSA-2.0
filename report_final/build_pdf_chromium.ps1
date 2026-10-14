# Render SALSA_2.0_DRDO_SAG_Final_Report.html to a print-ready A4 PDF.
#
# Drives headless Chrome through the DevTools Protocol rather than the
# --print-to-pdf command-line switch, because only the protocol accepts a
# custom footer template. That is what puts a clean, centred page number in the
# bottom margin instead of Chrome's default header/footer with the file URL and
# the date.
#
# The page size and margins come from the @page rule in the document's own CSS
# (preferCSSPageSize), so the PDF honours the stylesheet rather than Chrome's
# letter-size default.
#
# Run:  powershell -ExecutionPolicy Bypass -File report_final\build_pdf_chromium.ps1

$ErrorActionPreference = "Stop"

$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$html = Join-Path $dir "SALSA_2.0_DRDO_SAG_Final_Report.html"
$pdf  = Join-Path $dir "SALSA_2.0_DRDO_SAG_Final_Report.pdf"
if (-not (Test-Path $html)) { throw "Source not found: $html" }

# Chromium splits its command line on spaces and this project lives under a
# path that contains them, so stage the document somewhere without spaces.
$work = Join-Path $env:TEMP "salsa2_cdp"
Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $work | Out-Null
New-Item -ItemType Directory -Force (Join-Path $work "assets") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $work "SALSA_2.0_Report_Figures") | Out-Null
Copy-Item $html (Join-Path $work "report.html") -Force
Copy-Item (Join-Path $dir "assets\drdo_logo.png") (Join-Path $work "assets") -Force
Copy-Item (Join-Path $dir "SALSA_2.0_Report_Figures\*.png") (Join-Path $work "SALSA_2.0_Report_Figures") -Force

$browsers = @(
  "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
  "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
  "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe",
  "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
  "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe"
)
$browser = $null
foreach ($b in $browsers) { if (Test-Path $b) { $browser = $b; break } }
if ($null -eq $browser) { throw "No Chromium-based browser found." }
Write-Output "Renderer: $browser"

$port = 9333
$uri  = ([System.Uri](Join-Path $work "report.html")).AbsoluteUri

$args = @(
  "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
  "--no-default-browser-check", "--disable-extensions", "--disable-sync",
  "--disable-background-networking", "--mute-audio",
  "--user-data-dir=$work\prof",
  "--allow-file-access-from-files",
  "--remote-debugging-port=$port",
  "--remote-allow-origins=*",
  "about:blank"
)
$proc = Start-Process -FilePath $browser -ArgumentList $args -NoNewWindow -PassThru

function Stop-Browser {
    if ($null -ne $proc -and -not $proc.HasExited) { $proc.Kill(); $proc.WaitForExit(5000) }
}

try {
    # ---- wait for the debugging endpoint -------------------------------- #
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $null = Invoke-RestMethod "http://127.0.0.1:$port/json/version" -TimeoutSec 2
            $ready = $true; break
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { throw "Chrome DevTools endpoint did not come up on port $port." }
    Write-Output "DevTools ready."

    # ---- open the document in a new target ------------------------------ #
    $target = Invoke-RestMethod -Method Put -Uri "http://127.0.0.1:$port/json/new?$uri" -TimeoutSec 15
    $wsUrl  = $target.webSocketDebuggerUrl
    if ([string]::IsNullOrEmpty($wsUrl)) { throw "No webSocketDebuggerUrl returned." }
    Write-Output "Target opened."

    # ---- DevTools websocket --------------------------------------------- #
    $ws = New-Object System.Net.WebSockets.ClientWebSocket
    $ws.Options.SetRequestHeader("Origin", "http://127.0.0.1:$port")
    $ct = [System.Threading.CancellationToken]::None
    $ws.ConnectAsync([Uri]$wsUrl, $ct).GetAwaiter().GetResult()

    function Send-Cdp([int]$id, [string]$method, $params) {
        $msg = @{ id = $id; method = $method }
        if ($null -ne $params) { $msg.params = $params }
        $json  = $msg | ConvertTo-Json -Depth 10 -Compress
        $bytes = [Text.Encoding]::UTF8.GetBytes($json)
        $seg   = New-Object System.ArraySegment[byte] (, $bytes)
        $ws.SendAsync($seg, [System.Net.WebSockets.WebSocketMessageType]::Text, $true, $ct).GetAwaiter().GetResult()
    }

    function Receive-Cdp {
        $buf = New-Object byte[] 262144
        $seg = New-Object System.ArraySegment[byte] (, $buf)
        $sb  = New-Object System.Text.StringBuilder
        do {
            $r = $ws.ReceiveAsync($seg, $ct).GetAwaiter().GetResult()
            [void]$sb.Append([Text.Encoding]::UTF8.GetString($buf, 0, $r.Count))
        } while (-not $r.EndOfMessage)
        return $sb.ToString()
    }

    # Wait for the load event, then give the layout a moment to settle.
    Send-Cdp 1 "Page.enable" $null
    $deadline = (Get-Date).AddSeconds(90)
    $loaded = $false
    while ((Get-Date) -lt $deadline) {
        $raw = Receive-Cdp
        if ($raw -match '"method"\s*:\s*"Page.loadEventFired"') { $loaded = $true; break }
    }
    Write-Output "Load event: $loaded"
    Start-Sleep -Seconds 3

    # ---- print ----------------------------------------------------------- #
    $footer = '<div style="width:100%;margin:0 22mm 0 32mm;font-family:''Times New Roman'',serif;' +
              'font-size:10.5pt;color:#333333;text-align:center;">' +
              '<span class="pageNumber"></span></div>'

    $params = @{
        printBackground     = $true
        preferCSSPageSize   = $true      # honour the document's own @page rule
        displayHeaderFooter = $false
        headerTemplate      = '<span style="display:none"></span>'
        footerTemplate      = $footer
        marginBottom        = 0.6        # inches, room for the footer
        transferMode        = "ReturnAsBase64"
    }
    Send-Cdp 2 "Page.printToPDF" $params

    $data = $null
    $deadline = (Get-Date).AddSeconds(240)
    while ((Get-Date) -lt $deadline) {
        $raw = Receive-Cdp
        if ($raw -match '"id"\s*:\s*2') {
            $obj = $raw | ConvertFrom-Json
            if ($null -ne $obj.error) { throw "printToPDF failed: $($obj.error.message)" }
            $data = $obj.result.data
            break
        }
    }
    if ([string]::IsNullOrEmpty($data)) { throw "No PDF data returned." }

    [IO.File]::WriteAllBytes($pdf, [Convert]::FromBase64String($data))
    $ws.CloseAsync([System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure, "done", $ct).GetAwaiter().GetResult()
}
finally {
    Stop-Browser
}

$kb = [math]::Round((Get-Item $pdf).Length / 1KB, 1)
Write-Output "Wrote $pdf  ($kb KB)"
Write-Output "PDF_OK"
