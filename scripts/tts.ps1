param(
    [Parameter(Mandatory = $true)][string]$TextFile,
    [Parameter(Mandatory = $true)][string]$OutFile,
    [string]$Voice = "Microsoft Zira Desktop",
    [int]$Rate = 1
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice($Voice)
$synth.Rate = $Rate
$synth.Volume = 100
$synth.SetOutputToWaveFile($OutFile)
$synth.Speak((Get-Content -LiteralPath $TextFile -Raw -Encoding UTF8))
$synth.Dispose()
