param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$resolvedOutput = [IO.Path]::GetFullPath($OutputPath)
$outputDirectory = [IO.Path]::GetDirectoryName($resolvedOutput)
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}

$extension = [IO.Path]::GetExtension($resolvedInput).ToLowerInvariant()
$application = $null
$document = $null

try {
    if ($extension -eq ".docx") {
        $application = New-Object -ComObject Word.Application
        $application.Visible = $false
        $application.DisplayAlerts = 0
        $document = $application.Documents.Open(
            $resolvedInput,
            $false,
            $true
        )
        # 17 = wdExportFormatPDF
        $document.ExportAsFixedFormat($resolvedOutput, 17)
    }
    elseif ($extension -eq ".pptx") {
        $application = New-Object -ComObject PowerPoint.Application
        # Open(FileName, ReadOnly, Untitled, WithWindow)
        $document = $application.Presentations.Open(
            $resolvedInput,
            $true,
            $false,
            $false
        )
        # 32 = ppSaveAsPDF
        $document.SaveAs($resolvedOutput, 32)
    }
    else {
        throw "Unsupported Office extension: $extension"
    }
}
finally {
    if ($document -ne $null) {
        try { $document.Close() } catch {}
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($application -ne $null) {
        try { $application.Quit() } catch {}
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($application)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

if (-not (Test-Path -LiteralPath $resolvedOutput)) {
    throw "Office conversion did not create the PDF: $resolvedOutput"
}

Write-Output $resolvedOutput
