import platform, json, os, subprocess, sys
from pathlib import Path

root = Path(r'd:\friday')
root.mkdir(exist_ok=True)

info = {
    'platform': platform.platform(),
    'python_version': platform.python_version(),
    'python_arch': platform.architecture(),
    'machine': platform.machine(),
    'processor': platform.processor(),
    'release': platform.release(),
    'system': platform.system(),
    'win32_ver': platform.win32_ver(),
}

# Try native Windows WMI queries
cmds = [
    ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', 'Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory | ConvertTo-Json -Compress'],
    ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', 'Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress'],
    ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', 'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json -Compress'],
    ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', 'Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free, @{Name="Total";Expression={$_.Used + $_.Free}} | ConvertTo-Json -Compress'],
    ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', 'Get-WmiObject Win32_PnPEntity | Where-Object {$_.Name -match "microphone|audio|sound|wave|speaker|headphones"} | Select-Object Name,DeviceID | ConvertTo-Json -Compress'],
    ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', 'Get-WmiObject Win32_PnPEntity | Where-Object {$_.Name -match "camera|webcam|USB Video|Microsoft Camera"} | Select-Object Name,DeviceID | ConvertTo-Json -Compress'],
    ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', 'Get-Command ffmpeg -ErrorAction SilentlyContinue | Select-Object Source,Name | ConvertTo-Json -Compress'],
]

for cmd in cmds:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        info['cmd_' + '_'.join(cmd[0:2])] = { 'stdout': out.stdout.strip(), 'stderr': out.stderr.strip(), 'returncode': out.returncode }
    except Exception as e:
        info['cmd_error_' + cmd[0]] = repr(e)

# Torch / CUDA check
try:
    import torch
    info['torch_version'] = torch.__version__
    info['cuda_available'] = bool(torch.cuda.is_available())
    info['cuda_version'] = torch.version.cuda
    info['cuda_device_count'] = torch.cuda.device_count()
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_total_memory_mb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
except Exception as e:
    info['torch_error'] = repr(e)

path = root / 'machine_audit.json'
path.write_text(json.dumps(info, indent=2), encoding='utf-8')
print(path)
print(json.dumps({'python_version': info['python_version'], 'platform': info['platform'], 'machine': info['machine']}, indent=2))
