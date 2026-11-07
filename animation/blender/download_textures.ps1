$ErrorActionPreference = 'Stop'
$folder = Join-Path $PSScriptRoot 'textures'
New-Item -ItemType Directory -Path $folder -Force | Out-Null
$assets = @('dirt', 'concrete_floor', 'rock_01')
$maps = @('Diffuse', 'nor_gl', 'Rough')
$manifest = @()
foreach ($asset in $assets) {
    $api = Invoke-RestMethod -Uri "https://api.polyhaven.com/files/$asset" -Headers @{'User-Agent'='CodexBlenderScene/1.0'}
    foreach ($map in $maps) {
        $variants = $api.$map.'1k'
        if (-not $variants) { throw "Missing 1k $map for $asset" }
        $file = if ($variants.jpg) { $variants.jpg } elseif ($variants.png) { $variants.png } else { throw "No JPG/PNG for $asset $map" }
        $uri = [uri]$file.url
        if ($uri.Host -ne 'dl.polyhaven.org') { throw "Unexpected host: $uri" }
        $extension = [IO.Path]::GetExtension($uri.AbsolutePath)
        $target = Join-Path $folder "${asset}_${map}_1k${extension}"
        Invoke-WebRequest -Uri $uri -OutFile $target -Headers @{'User-Agent'='CodexBlenderScene/1.0'}
        $size = (Get-Item -LiteralPath $target).Length
        if ($size -lt 10000) { throw "Incomplete download: $target" }
        $manifest += [pscustomobject]@{asset=$asset; map=$map; file=(Split-Path $target -Leaf); bytes=$size; url=$file.url; source="https://polyhaven.com/a/$asset"; license='CC0'}
        Write-Output "Downloaded $(Split-Path $target -Leaf) ($size bytes)"
    }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $folder 'asset_manifest.json') -Encoding utf8
