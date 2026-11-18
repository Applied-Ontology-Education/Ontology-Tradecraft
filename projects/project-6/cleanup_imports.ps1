<#
Build a lean, SELF-CONTAINED TTL using ROBOT locality module extraction.
NEVER removes ANY original assertions other than owl:imports.
Requires: Python 3.10+, rdflib, ROBOT (robot on PATH), Java, working imports or an XML catalog.
Example:
 .\cleanup_imports.ps1 -InputFile '.\Ontology_Intel_Analysis(20261111-182522).ttl' -OutputFile '.\Ontology_Intel_Analysis_LEAN.ttl'
Or use -CatalogFile '.\catalog-v001.xml' for local imported ontology files.
#>
param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [Parameter(Mandatory=$true)][string]$OutputFile,
    [string]$CatalogFile,
    [ValidateSet('BOT','STAR')][string]$Method='BOT',
    [string]$Python='python',
    [string]$Robot='robot'
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$InputFile = (Resolve-Path -LiteralPath $InputFile).Path
$OutputFile = [System.IO.Path]::GetFullPath($OutputFile)
if ($InputFile -eq $OutputFile) { throw 'Input and output must be different files.' }
$Script = Join-Path $PSScriptRoot 'build_module.py'
$work = Join-Path ([System.IO.Path]::GetDirectoryName($OutputFile)) '.robot_lean_work'
New-Item -ItemType Directory -Force -Path $work | Out-Null
$seedFile = Join-Path $work 'preserved_terms.txt'
$merged = Join-Path $work 'merged_import_closure.ttl'
$module = Join-Path $work 'import_module.ttl'
$report = [System.IO.Path]::ChangeExtension($OutputFile, '.cleanup_report.txt')

function Invoke-Checked([string]$program,[string[]]$arguments) {
    Write-Host ('> ' + $program + ' ' + ($arguments -join ' '))
    & $program @arguments
    if ($LASTEXITCODE -ne 0) { throw "$program failed with exit code $LASTEXITCODE" }
}

Invoke-Checked $Python @($Script,'prepare','--input',$InputFile,'--seeds',$seedFile)
$mergeArgs = @('merge','--input',$InputFile)
if ($CatalogFile) {
    $CatalogFile = (Resolve-Path -LiteralPath $CatalogFile).Path
    $mergeArgs += @('--catalog',$CatalogFile)
}
$mergeArgs += @('--output',$merged)
Invoke-Checked $Robot $mergeArgs
# BOT retains imported superclasses and relevant relationships, suitable for
# seeing meaningful hierarchy in Protege. STAR normally produces a smaller module.
Invoke-Checked $Robot @('extract','--input',$merged,'--method',$Method,'--term-file',$seedFile,'--imports','exclude','--individuals','exclude','--output',$module)
Invoke-Checked $Python @($Script,'finish','--input',$InputFile,'--module',$module,'--output',$OutputFile,'--report',$report)
Write-Host "Lean ontology: $OutputFile"
Write-Host "Report: $report"
Write-Host 'Keep the .robot_lean_work folder for auditing until you validate in Protege.'
