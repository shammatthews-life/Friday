import platform, json
from pathlib import Path

info = {
    'system': platform.system(),
    'release': platform.release(),
    'version': platform.version(),
    'platform': platform.platform(),
    'machine': platform.machine(),
    'python_version': platform.python_version(),
    'python_implementation': platform.python_implementation(),
    'architecture': platform.architecture(),
    'win32_ver': platform.win32_ver(),
}
Path(r'd:\friday\machine_probe.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
print(json.dumps(info, indent=2))
