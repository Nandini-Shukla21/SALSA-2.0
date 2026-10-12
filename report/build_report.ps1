# Build the SALSA 2.0 internship report as .docx and .pdf from salsa2_report.html.
#
# Uses Microsoft Word through COM automation. No third-party converter is
# installed in this environment, and none is required by this script.
#
# What it does beyond a plain conversion:
#   - A4 page setup with the margins of the reference report
#   - Times New Roman 12 pt, 1.5 line spacing, justified body text
#   - embeds the linked figures so the .docx is self-contained
#   - applies Word's Caption style to figure and table captions
#   - inserts a real Table of Contents, List of Figures and List of Tables as
#     Word fields, so their page numbers are correct and stay updatable
#   - splits front matter and body into two sections, with roman numerals in
#     the front matter and arabic numerals restarting at 1 in the body
#
# Run:  powershell -ExecutionPolicy Bypass -File report\build_report.ps1

$ErrorActionPreference = "Stop"

$reportDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$htmlPath  = Join-Path $reportDir "salsa2_report.html"

# Word is driven headlessly, and saving straight into a cloud-synced folder can
# block on the sync client. Build in a plain local directory, then copy the
# finished files back beside the source.
$workDir   = Join-Path $env:TEMP "salsa2_report_build"
if (-not (Test-Path $workDir)) { New-Item -ItemType Directory -Force $workDir | Out-Null }

$docxName  = "SALSA_2.0_Internship_Report.docx"
$pdfName   = "SALSA_2.0_Internship_Report.pdf"
$docxPath  = Join-Path $workDir $docxName
$pdfPath   = Join-Path $workDir $pdfName
$finalDocx = Join-Path $reportDir $docxName
$finalPdf  = Join-Path $reportDir $pdfName

if (-not (Test-Path $htmlPath)) { throw "Source not found: $htmlPath" }

# Work on a local copy of the source, with the figures beside it, so that Word
# never touches the synced folder while the document is open.
Copy-Item $htmlPath (Join-Path $workDir "salsa2_report.html") -Force
$figSrc = Join-Path $reportDir "figures"
$figDst = Join-Path $workDir "figures"
if (-not (Test-Path $figDst)) { New-Item -ItemType Directory -Force $figDst | Out-Null }
Copy-Item (Join-Path $figSrc "*.png") $figDst -Force
$htmlPath = Join-Path $workDir "salsa2_report.html"

# ---- Word constants ------------------------------------------------------ #
$wdFormatDocumentDefault = 16
$wdExportFormatPDF       = 17
$wdPaperA4               = 7
$wdSectionBreakNextPage  = 2
$wdHeaderFooterPrimary   = 1
$wdFieldPage             = 33
$wdPageNumberStyleArabic = 0
$wdPageNumberStyleRoman  = 4   # lowercase roman
$wdAlignParagraphCenter  = 1
$wdLineSpaceMultiple     = 5
$wdStory                 = 6
$wdFindStop              = 0

Write-Output "Opening Word..."
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
# Every style or caption edit otherwise triggers a background repagination of
# the whole document. On a 40-page report with embedded figures that dominates
# the build time, so pagination is suspended here and performed once, below.
try { $word.Options.Pagination = $false } catch { }
try { $word.ScreenUpdating = $false } catch { }

try {
    $doc = $word.Documents.Open($htmlPath, [ref]$false, [ref]$false)

    # ---- 1. Page setup --------------------------------------------------- #
    foreach ($sec in $doc.Sections) {
        $ps = $sec.PageSetup
        $ps.PaperSize    = $wdPaperA4
        $ps.TopMargin    = $word.InchesToPoints(1.0)
        $ps.BottomMargin = $word.InchesToPoints(1.0)
        $ps.LeftMargin   = $word.InchesToPoints(1.25)
        $ps.RightMargin  = $word.InchesToPoints(1.0)
        $ps.Gutter       = 0
        $ps.FooterDistance = $word.InchesToPoints(0.5)
    }

    # ---- 2. Base typography ---------------------------------------------- #
    $normal = $doc.Styles.Item("Normal")
    $normal.Font.Name = "Times New Roman"
    $normal.Font.Size = 12
    $normal.ParagraphFormat.LineSpacingRule = $wdLineSpaceMultiple
    $normal.ParagraphFormat.LineSpacing = $word.LinesToPoints(1.5)
    $normal.ParagraphFormat.SpaceAfter = 6

    $headingSizes = [ordered]@{ "Heading 1" = 16.0; "Heading 2" = 13.5; "Heading 3" = 12.5 }
    foreach ($name in $headingSizes.Keys) {
        $st = $doc.Styles.Item($name)
        $st.Font.Name = "Times New Roman"
        $st.Font.Size = [single]$headingSizes[$name]
        $st.Font.Bold = $true
        $st.Font.Color = 0            # black, not Word's default blue
        $st.ParagraphFormat.SpaceBefore = 12
        $st.ParagraphFormat.SpaceAfter = 8
    }

    # ---- 3. Embed the figures so the .docx is self-contained -------------- #
    $embedded = 0
    foreach ($shape in $doc.InlineShapes) {
        if ($shape.Type -eq 3 -or $null -ne $shape.LinkFormat) {
            try {
                $shape.LinkFormat.SavePictureWithDocument = $true
                $shape.LinkFormat.BreakLink()
                $embedded++
            } catch { }
        }
    }
    Write-Output "Embedded $embedded linked image(s)."

    # ---- 4. Caption style on figure and table captions -------------------- #
    # The List of Figures / List of Tables fields collect paragraphs carrying
    # Word's built-in Caption style, so the captions must be marked as such.
    # wdStyleCaption = -35; resolve it by id so the script is locale-independent.
    $captionStyle = $null
    try { $captionStyle = $doc.Styles.Item(-35) } catch { $captionStyle = $null }
    if ($null -eq $captionStyle) { $captionStyle = "Caption" }

    $captions = 0
    foreach ($para in $doc.Paragraphs) {
        $t = $para.Range.Text
        if ($t -match '^\s*(Figure|Table)\s+[A-C0-9]+\.[0-9]+:') {
            $para.Style = $captionStyle
            $para.Range.Font.Name = "Times New Roman"
            $para.Range.Font.Size = 10.5
            $para.Range.Font.Italic = $false
            $para.Range.Font.Color = 0
            $captions++
        }
    }
    Write-Output "Styled $captions caption paragraph(s)."

    # ---- 5. Section break between front matter and body ------------------- #
    $broke = $false
    foreach ($para in $doc.Paragraphs) {
        if ($para.Range.Text -match '\[\[SECTIONBREAK\]\]') {
            $r = $para.Range
            $r.Text = ""
            $r.InsertBreak($wdSectionBreakNextPage)
            $broke = $true
            break
        }
    }
    Write-Output "Section break inserted: $broke  (sections: $($doc.Sections.Count))"

    # ---- 6. Page numbering ------------------------------------------------ #
    # Front matter: lowercase roman. Body: arabic, restarting at 1.
    for ($i = 1; $i -le $doc.Sections.Count; $i++) {
        $sec = $doc.Sections.Item($i)
        $footer = $sec.Footers.Item($wdHeaderFooterPrimary)
        if ($i -gt 1) { $footer.LinkToPrevious = $false }
        $footer.Range.Text = ""
        $footer.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
        $footer.Range.Font.Name = "Times New Roman"
        $footer.Range.Font.Size = 11
        $null = $footer.Range.Fields.Add($footer.Range, $wdFieldPage)

        $pn = $sec.Footers.Item($wdHeaderFooterPrimary).PageNumbers
        if ($i -eq 1) {
            $pn.NumberStyle = $wdPageNumberStyleRoman
            $pn.RestartNumberingAtSection = $true
            $pn.StartingNumber = 1
        } else {
            $pn.NumberStyle = $wdPageNumberStyleArabic
            $pn.RestartNumberingAtSection = $true
            $pn.StartingNumber = 1
        }
    }

    Write-Output "Page numbering set."

    # ---- 7. Remaining placeholders ---------------------------------------- #
    # Done BEFORE the reference fields are inserted: a find-and-replace sweep
    # over a document that already contains a TOC forces Word to repaginate
    # repeatedly, which is very slow on a document with embedded figures.
    $subs = [ordered]@{
        "[[ROLL]]"      = "[ROLL NUMBER - TO BE SUPPLIED]"
        "[[DATE]]"      = "[DATE - TO BE SUPPLIED]"
        "[[MONTHYEAR]]" = "[MONTH YEAR OF SUBMISSION - TO BE CONFIRMED]"
        "[[LOGO]]"      = "[UNIVERSITY LOGO - TO BE INSERTED]"
    }
    foreach ($key in $subs.Keys) {
        $find = $doc.Content.Find
        $find.ClearFormatting()
        $find.Replacement.ClearFormatting()
        $null = $find.Execute($key, $false, $false, $false, $false, $false, $true,
                              $wdFindStop, $false, $subs[$key], 2)
    }
    Write-Output "Placeholders substituted."

    # ---- 8. Table of Contents, List of Figures, List of Tables ------------ #
    function Get-PlaceholderRange($doc, $token) {
        foreach ($para in $doc.Paragraphs) {
            if ($para.Range.Text -match [regex]::Escape($token)) { return $para.Range }
        }
        return $null
    }

    $r = Get-PlaceholderRange $doc "[[TOC]]"
    if ($null -ne $r) {
        $r.Text = ""
        $null = $doc.TablesOfContents.Add($r, $true, 1, 3, $false, "", $true, $true, "", $true, $false, $true)
        Write-Output "Table of Contents inserted."
    }

    $r = Get-PlaceholderRange $doc "[[LOF]]"
    if ($null -ne $r) {
        $r.Text = ""
        $null = $doc.TablesOfFigures.Add($r, "Figure", $true, $false, 1, 3, $false, "", $true, $true, "", $true, $false)
        Write-Output "List of Figures inserted."
    }

    $r = Get-PlaceholderRange $doc "[[LOT]]"
    if ($null -ne $r) {
        $r.Text = ""
        $null = $doc.TablesOfFigures.Add($r, "Table", $true, $false, 1, 3, $false, "", $true, $true, "", $true, $false)
        Write-Output "List of Tables inserted."
    }

    # ---- 9. Paginate once, then update the reference fields --------------- #
    # Pagination must be restored before the lists are updated, otherwise their
    # page numbers would be computed against a stale layout.
    try { $word.Options.Pagination = $true } catch { }
    $doc.Repaginate()
    Write-Output "Repaginated: $($doc.ComputeStatistics(2)) pages."

    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    Write-Output "Table of Contents updated."
    foreach ($tof in $doc.TablesOfFigures)  { $tof.Update() }
    Write-Output "Lists of figures and tables updated."

    # ---- 10. Save --------------------------------------------------------- #
    $doc.SaveAs2($docxPath, $wdFormatDocumentDefault)
    Write-Output "Wrote $docxPath"
    $doc.ExportAsFixedFormat($pdfPath, $wdExportFormatPDF)
    Write-Output "Wrote $pdfPath"
    Write-Output "Pages: $($doc.ComputeStatistics(2))"
    Write-Output "Words: $($doc.ComputeStatistics(0))"

    $doc.Close(0)
}
finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
}

# ---- 11. Copy the finished files back beside the source ------------------ #
Copy-Item $docxPath $finalDocx -Force
Copy-Item $pdfPath  $finalPdf  -Force
Write-Output "Copied to $finalDocx"
Write-Output "Copied to $finalPdf"
Write-Output "BUILD_OK"
