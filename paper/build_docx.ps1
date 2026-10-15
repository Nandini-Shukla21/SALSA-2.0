# Produce an editable Word version of the manuscript from the same HTML the
# PDF is rendered from.
#
# Word reflows the two-column layout and will not reproduce the figure
# placement or the equation rendering faithfully.  The DOCX is an editing
# convenience for track-changes review; the PDF is the typeset artefact.
#
# Run:  powershell -ExecutionPolicy Bypass -File paper\build_docx.ps1

$ErrorActionPreference = "Stop"

$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$html = Join-Path $dir "_word_source.html"
$docx = Join-Path $dir "Modified_SALSA_Research_Paper.docx"
if (-not (Test-Path $html)) { throw "Source not found: $html" }

Get-Process WINWORD -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    Write-Output "Opening the HTML in Word..."
    $doc = $word.Documents.Open($html, [ref]$false, [ref]$false)

    # Repagination on every edit is the single biggest cost in Word
    # automation; turn it off, make the changes, then paginate once.
    $word.Options.Pagination = $false

    # Linked images must be embedded or the DOCX breaks once it is moved.
    $embedded = 0
    foreach ($shape in @($doc.InlineShapes)) {
        if ($null -ne $shape.LinkFormat) {
            $shape.LinkFormat.SavePictureWithDocument = $true
            $shape.LinkFormat.BreakLink()
            $embedded++
        }
    }
    Write-Output "Embedded $embedded linked image(s)."

    # A4 with the same margins as the PDF.
    $ps = $doc.PageSetup
    $ps.PageWidth  = $word.CentimetersToPoints(21.0)
    $ps.PageHeight = $word.CentimetersToPoints(29.7)
    $ps.TopMargin    = $word.CentimetersToPoints(1.7)
    $ps.BottomMargin = $word.CentimetersToPoints(1.8)
    $ps.LeftMargin   = $word.CentimetersToPoints(1.5)
    $ps.RightMargin  = $word.CentimetersToPoints(1.5)

    # Page numbers in the footer.
    $footer = $doc.Sections.Item(1).Footers.Item(1)
    $footer.PageNumbers.Add(2, $true) | Out-Null   # wdAlignPageNumberCenter

    $word.Options.Pagination = $true
    $doc.Repaginate()
    Write-Output "Repaginated: $($doc.ComputeStatistics(2)) pages."

    $doc.SaveAs2($docx, 16)        # wdFormatDocumentDefault = .docx
    $pages = $doc.ComputeStatistics(2)
    $words = $doc.ComputeStatistics(0)
    $doc.Close([ref]$false)
    Write-Output "Saved $docx  ($pages pages, $words words)."
}
finally {
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    Get-Process WINWORD -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

Write-Output "DOCX_OK"
