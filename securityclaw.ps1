[CmdletBinding()]
param(
    [ValidateSet('start', 'local', 'docker', 'stop', 'status', 'chat', 'skills', 'logs')]
    [string]$Action = 'start',
    [Alias('h')]
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OpenSearchContainer = if ($env:SECURITYCLAW_OPENSEARCH_CONTAINER) { $env:SECURITYCLAW_OPENSEARCH_CONTAINER } else { 'securityclaw-opensearch' }
$OpenSearchVolume = if ($env:SECURITYCLAW_OPENSEARCH_VOLUME) { $env:SECURITYCLAW_OPENSEARCH_VOLUME } else { 'securityclaw-opensearch-data' }
$OpenSearchImage = if ($env:SECURITYCLAW_OPENSEARCH_IMAGE) { $env:SECURITYCLAW_OPENSEARCH_IMAGE } else { 'opensearchproject/opensearch:2' }
$OllamaUrl = if ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL.TrimEnd('/') } else { 'http://127.0.0.1:11434' }
$AppUrl = if ($env:SECURITYCLAW_URL) { $env:SECURITYCLAW_URL } else { 'http://127.0.0.1:7799' }
$Python = Join-Path $Root '.venv\Scripts\python.exe'

if ($Help) {
    @'
SecurityClaw native Windows launcher

Usage: .\securityclaw.ps1 [-Action] <start|local|docker|stop|status|chat|skills|logs>
       .\securityclaw.ps1 -h

  start, local  Run with the local Windows virtual environment.
  docker        Run the application through Docker Compose.
  stop          Stop Compose and the managed OpenSearch container.
  status        Show application, Ollama, and OpenSearch availability.
  chat          Open terminal chat.
  skills        List skills compatible with Windows.
  logs          Follow OpenSearch logs.

Environment overrides match securityclaw.sh. The launcher never enables a
container restart-at-boot policy.
'@ | Write-Host
    exit 0
}

function Test-Command([string]$Name) { return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }
function Wait-Url([string]$Name, [string]$Url, [int]$Attempts = 45) {
    for ($i = 0; $i -lt $Attempts; $i++) {
        try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri $Url | Out-Null; Write-Host "$Name is ready."; return }
        catch { Start-Sleep -Seconds 2 }
    }
    throw "$Name did not become ready at $Url"
}
function Invoke-Compose([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments) {
    & docker compose version *> $null
    if ($LASTEXITCODE -eq 0) { & docker compose @Arguments; return }
    if (Test-Command 'docker-compose') { & docker-compose @Arguments; return }
    throw 'Docker Compose is required for container mode.'
}
function Ensure-Docker {
    if (-not (Test-Command 'docker')) { throw 'Docker Desktop is required.' }
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        $desktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
        if (Test-Path $desktop) { Start-Process $desktop | Out-Null }
        for ($i = 0; $i -lt 45; $i++) { Start-Sleep 2; & docker info *> $null; if ($LASTEXITCODE -eq 0) { return } }
        throw 'Docker Desktop is not available.'
    }
}
function Ensure-OpenSearch {
    Ensure-Docker
    & docker container inspect $OpenSearchContainer *> $null
    if ($LASTEXITCODE -eq 0) {
        & docker update --restart=no $OpenSearchContainer *> $null
        $running = & docker inspect -f '{{.State.Running}}' $OpenSearchContainer
        if ($running -ne 'true') { & docker start $OpenSearchContainer *> $null }
    } else {
        Write-Host 'Creating the minimal OpenSearch dependency...'
        & docker volume create $OpenSearchVolume *> $null
        & docker run -d --name $OpenSearchContainer --restart=no `
            -p '127.0.0.1:9200:9200' -p '127.0.0.1:9600:9600' `
            -e 'discovery.type=single-node' -e 'DISABLE_SECURITY_PLUGIN=true' `
            -e 'OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g' `
            -v "${OpenSearchVolume}:/usr/share/opensearch/data" $OpenSearchImage *> $null
    }
    Wait-Url 'OpenSearch' 'http://127.0.0.1:9200' 60
}
function Check-Configuration {
    if (-not (Test-Path (Join-Path $Root 'config.yaml'))) { throw 'config.yaml is missing. Run: .\.venv\Scripts\python.exe main.py onboard' }
}
function Check-Ollama { Wait-Url 'Ollama' "$OllamaUrl/api/tags" 15 }

Set-Location $Root
switch ($Action) {
    { $_ -in @('start', 'local') } {
        Check-Configuration
        if (-not (Test-Path $Python)) { throw '.venv is missing. Create it and install requirements.txt first.' }
        Ensure-OpenSearch
        Check-Ollama
        Write-Host "SecurityClaw is available at $AppUrl (Ctrl+C stops the application)."
        & $Python (Join-Path $Root 'main.py') service
    }
    'docker' {
        Check-Configuration
        Ensure-OpenSearch
        Check-Ollama
        Invoke-Compose up --build
    }
    'stop' {
        if (Test-Command 'docker') {
            try { Invoke-Compose down } catch { }
            & docker stop $OpenSearchContainer *> $null
        }
        Write-Host 'SecurityClaw dependencies stopped. No restart-at-boot policy was enabled.'
    }
    'status' {
        try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 $AppUrl | Out-Null; $app = 'online' } catch { $app = 'stopped' }
        try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "$OllamaUrl/api/tags" | Out-Null; $ollama = 'online' } catch { $ollama = 'unavailable' }
        Write-Host "Application        $app"
        Write-Host "Ollama            $ollama"
        if (Test-Command 'docker') { & docker inspect -f 'OpenSearch         {{.State.Status}} (restart={{.HostConfig.RestartPolicy.Name}})' $OpenSearchContainer }
    }
    'chat' { Check-Configuration; & $Python (Join-Path $Root 'main.py') chat }
    'skills' { Check-Configuration; & $Python (Join-Path $Root 'main.py') list-skills }
    'logs' { Ensure-Docker; & docker logs --tail=200 -f $OpenSearchContainer }
}
