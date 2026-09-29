$ErrorActionPreference = 'Continue'
$root = 'd:\friday'
$machine = [ordered]@{}
$machine['os_version'] = [System.Environment]::OSVersion.VersionString
$machine['os_platform'] = [System.Environment]::OSVersion.Platform.ToString()
$machine['python_version'] = (python --version 2>&1 | Select-Object -First 1)
$machine['machine_name'] = [System.Environment]::MachineName
$machine['user_name'] = [System.Environment]::UserName
$machine['processor'] = (Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name -First 1)
$machine['logical_cores'] = (Get-CimInstance Win32_Processor | Select-Object -ExpandProperty NumberOfLogicalProcessors -First 1)
$machine['physical_cores'] = (Get-CimInstance Win32_Processor | Select-Object -ExpandProperty NumberOfCores -First 1)
$machine['ram_mb'] = [math]::Round((Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory) / 1MB, 2)
$machine['gpu'] = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name -First 1)
$machine['gpu_driver'] = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty DriverVersion -First 1)
$machine['gpu_vram_mb'] = [math]::Round((Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty AdapterRAM -First 1) / 1MB, 2)
$machine['storage'] = @(Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{Name='Used';Expression={$_.Used}}, @{Name='Free';Expression={$_.Free}}, @{Name='Total';Expression={$_.Used + $_.Free}} | ConvertTo-Json -Compress)
$machine['ffmpeg'] = (Get-Command ffmpeg -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)
$machine['audio_devices'] = @(Get-WmiObject Win32_PnPEntity | Where-Object { $_.Name -match 'microphone|audio|sound|wave|speaker|headphones' } | Select-Object -ExpandProperty Name)
$machine['camera_devices'] = @(Get-WmiObject Win32_PnPEntity | Where-Object { $_.Name -match 'camera|webcam|USB Video|Microsoft Camera' } | Select-Object -ExpandProperty Name)
try { $torch = python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1; $machine['torch'] = $torch } catch { $machine['torch'] = 'not installed' }
$machine['cuda_available'] = $false
try { $cuda = python -c "import torch; print(torch.cuda.is_available())" 2>&1; if ($cuda -match 'True') { $machine['cuda_available'] = $true } } catch { }
$output = $machine | ConvertTo-Json -Depth 8
Set-Content -Path 'd:\friday\machine_report.json' -Value $output
Write-Host 'WROTE d:\friday\machine_report.json'
