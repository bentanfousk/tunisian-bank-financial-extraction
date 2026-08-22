$ErrorActionPreference = "Stop"

# Exact models declared in configs/models.yaml.
$expectedModels = @(
    "qwen3:4b-instruct-2507-q4_K_M",
    "ministral-3:3b-instruct-2512-q4_K_M",
    "gemma3:4b-it-q4_K_M",
    "mistral:7b-instruct-v0.3-q4_K_M"
)

# Retrieve the locally installed artifacts from Ollama.
$installedModels = (
    Invoke-RestMethod -Uri "http://localhost:11434/api/tags"
).models

$lockedModels = @()

foreach ($expectedName in $expectedModels) {
    $model = $installedModels |
        Where-Object { $_.name -eq $expectedName } |
        Select-Object -First 1

    if (-not $model) {
        throw "Required model is not installed: $expectedName"
    }

    if ($model.details.quantization_level -ne "Q4_K_M") {
        throw (
            "Unexpected quantization for ${expectedName}: " +
            $model.details.quantization_level
        )
    }

    $lockedModels += [ordered]@{
        name           = $model.name
        digest         = $model.digest
        size_bytes     = $model.size
        modified_at    = $model.modified_at
        format         = $model.details.format
        family         = $model.details.family
        parameter_size = $model.details.parameter_size
        quantization   = $model.details.quantization_level
    }
}

$lock = [ordered]@{
    schema_version = 1
    generated_at   = (Get-Date).ToString("o")
    ollama_version = ((ollama --version) -join " ")
    models         = $lockedModels
}

$outputPath = ".\configs\models.lock.json"

$lock |
    ConvertTo-Json -Depth 8 |
    Set-Content -Path $outputPath -Encoding UTF8

Write-Host ""
Write-Host "Created: $outputPath"
Write-Host "Locked models: $($lockedModels.Count)"