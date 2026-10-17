# Render selected pages of the finished PDF to PNG so the layout can be checked
# by eye. Uses the Windows runtime PDF rasteriser (Windows.Data.Pdf), which is
# part of the OS, so no additional package is required.
#
# Run:  powershell -ExecutionPolicy Bypass -File report_final\preview_pdf_pages.ps1 1,5,8,20,34

param([string]$Pages = "1,5,8,20,34,59")

$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pdf = Join-Path $dir "Modified_SALSA_Research_Paper.pdf"
$out = Join-Path $env:TEMP "salsa2_paper_preview"
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Force $out | Out-Null }

Add-Type -AssemblyName System.Runtime.WindowsRuntime

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
                   $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]

function Await($op, $type) {
    $m = $asTaskGeneric.MakeGenericMethod($type)
    $t = $m.Invoke($null, @($op))
    $t.Wait(-1) | Out-Null
    $t.Result
}
$asTaskAction = ([System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
                   $_.GetParameters()[0].ParameterType.FullName -eq 'Windows.Foundation.IAsyncAction' })[0]

function AwaitAction($op) {
    $t = $asTaskAction.Invoke($null, @($op))
    $t.Wait(-1) | Out-Null
}

[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime] | Out-Null

$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($pdf)) ([Windows.Storage.StorageFile])
$doc  = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])
Write-Output "PDF page count: $($doc.PageCount)"

$pageList = $Pages -split ',' | ForEach-Object { [int]$_.Trim() }
foreach ($n in $pageList) {
    if ($n -lt 1 -or $n -gt $doc.PageCount) { Write-Output "skip page $n (out of range)"; continue }
    $page = $doc.GetPage($n - 1)
    $stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
    $opts = New-Object Windows.Data.Pdf.PdfPageRenderOptions
    $opts.DestinationWidth = 1240          # ~150 dpi at A4 width
    AwaitAction ($page.RenderToStreamAsync($stream, $opts))

    $size = [uint32]$stream.Size
    $null = $stream.Seek(0)
    $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
    Await ($reader.LoadAsync($size)) ([uint32]) | Out-Null
    $bytes = New-Object byte[] $size
    $reader.ReadBytes($bytes)
    $reader.Dispose()
    $stream.Dispose()
    $page.Dispose()

    $path = Join-Path $out ("page_{0:d2}.png" -f $n)
    [IO.File]::WriteAllBytes($path, $bytes)
    Write-Output "wrote $path ($([math]::Round($bytes.Length/1KB,1)) KB)"
}
Write-Output "PREVIEW_OK"
