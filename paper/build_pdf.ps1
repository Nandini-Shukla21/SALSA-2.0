# Render Modified_SALSA_Research_Paper.html to a print-ready A4 PDF.
#
# Drives headless Chrome through the DevTools Protocol rather than the
# --print-to-pdf switch, because the protocol honours the document's own
# @page rule (preferCSSPageSize), including its margin boxes, so the page
# geometry and the footer page numbers come from the stylesheet.
#
# Run:  powershell -ExecutionPolicy Bypass -File paper\build_pdf.ps1

$ErrorActionPreference = "Stop"

$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$html = Join-Path $dir "Modified_SALSA_Research_Paper.html"
$pdf  = Join-Path $dir "Modified_SALSA_Research_Paper.pdf"
if (-not (Test-Path $html)) { throw "Source not found: $html" }

# Chromium splits its command line on spaces and this project lives under a
# path that contains them, so stage the document somewhere without spaces.
$work = Join-Path $env:TEMP "salsa2_paper_cdp"
Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $work | Out-Null
New-Item -ItemType Directory -Force (Join-Path $work "Modified_SALSA_Figures") | Out-Null
Copy-Item $html (Join-Path $work "paper.html") -Force
Copy-Item (Join-Path $dir "Modified_SALSA_Figures\*.png") `
          (Join-Path $work "Modified_SALSA_Figures") -Force

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

$port = 9334
$uri  = ([System.Uri](Join-Path $work "paper.html")).AbsoluteUri

$chromeArgs = @(
  "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
  "--no-default-browser-check", "--disable-extensions", "--disable-sync",
  "--disable-background-networking", "--mute-audio",
  "--user-data-dir=$work\prof",
  "--allow-file-access-from-files",
  "--remote-debugging-port=$port",
  "--remote-allow-origins=*",
  "about:blank"
)
$proc = Start-Process -FilePath $browser -ArgumentList $chromeArgs -NoNewWindow -PassThru

function Stop-Browser {
    if ($null -ne $proc -and -not $proc.HasExited) { $proc.Kill(); $proc.WaitForExit(5000) }
}

try {
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $null = Invoke-RestMethod "http://127.0.0.1:$port/json/version" -TimeoutSec 2
            $ready = $true; break
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { throw "Chrome DevTools endpoint did not come up on port $port." }
    Write-Output "DevTools ready."

    # Open a BLANK target and navigate only after Page events are enabled.
    # Opening the document directly races the load event against our
    # subscription: if the page finishes loading first the event is never
    # delivered and the wait below blocks forever.
    $blank  = [System.Uri]::EscapeDataString("about:blank")
    $target = Invoke-RestMethod -Method Put -Uri "http://127.0.0.1:$port/json/new?$blank" -TimeoutSec 15
    $wsUrl  = $target.webSocketDebuggerUrl
    if ([string]::IsNullOrEmpty($wsUrl)) { throw "No webSocketDebuggerUrl returned." }
    Write-Output "Blank target opened."

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

    # Every receive is bounded, so a message that never arrives cannot wedge
    # the script: the wait loops below re-check their own deadline instead.
    function Receive-Cdp([int]$timeoutSeconds = 30) {
        $buf = New-Object byte[] 262144
        $seg = New-Object System.ArraySegment[byte] (, $buf)
        $sb  = New-Object System.Text.StringBuilder
        $cts = New-Object System.Threading.CancellationTokenSource
        $cts.CancelAfter([TimeSpan]::FromSeconds($timeoutSeconds))
        try {
            do {
                $r = $ws.ReceiveAsync($seg, $cts.Token).GetAwaiter().GetResult()
                [void]$sb.Append([Text.Encoding]::UTF8.GetString($buf, 0, $r.Count))
            } while (-not $r.EndOfMessage)
        } catch { return $null }
        finally { $cts.Dispose() }
        return $sb.ToString()
    }

    Send-Cdp 1 "Page.enable" $null
    Send-Cdp 3 "Page.navigate" @{ url = $uri }
    $deadline = (Get-Date).AddSeconds(120)
    $loaded = $false
    while ((Get-Date) -lt $deadline) {
        $raw = Receive-Cdp 10
        if ($null -eq $raw) { continue }
        if ($raw -match '"method"\s*:\s*"Page.loadEventFired"') { $loaded = $true; break }
    }
    Write-Output "Load event: $loaded"
    if (-not $loaded) { throw "Page never reported a load event." }
    Start-Sleep -Seconds 3

    $params = @{
        printBackground     = $true
        preferCSSPageSize   = $true      # honour the document's own @page rule
        displayHeaderFooter = $false     # the stylesheet supplies the footer
        transferMode        = "ReturnAsBase64"
    }
    Send-Cdp 2 "Page.printToPDF" $params

    $data = $null
    $deadline = (Get-Date).AddSeconds(240)
    while ((Get-Date) -lt $deadline) {
        $raw = Receive-Cdp 30
        if ($null -eq $raw) { continue }
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
