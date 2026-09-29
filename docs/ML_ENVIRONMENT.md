# ML ENVIRONMENT

## Windows and Python

- OS: Windows 11
- Python: 3.14.5
- Architecture: 64-bit AMD64
- Virtual environment: d:\friday\.venv

## NVIDIA driver and GPU

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- Dedicated VRAM: 4 GB
- Integrated GPU: AMD Radeon 740M
- Driver version: 32.0.15.9282
- nvidia-smi driver: 592.82 (GPU detected)
- nvidia-smi status: verified on the machine
- CUDA compatibility reported by driver: GPU/driver visible, but PyTorch CUDA runtime is not verified

## PyTorch and torchvision

The current venv contains importable CPU wheels. The official CUDA 12.8 replacement was confirmed compatible for Python 3.14, but the single installation attempt was stopped during the 2.8 GB torch wheel download after only about 0.3 GB transferred at approximately 1.5 MB/s.

- PyTorch: 2.14.0+cpu (import verified)
- torchvision: 0.29.0+cpu (import verified)
- CUDA build: not available (`torch.version.cuda` is `None`)
- `torch.cuda.is_available()`: `False`
- CUDA tensor operation: failed with `AssertionError: Torch not compiled with CUDA enabled`
- Python compatibility: Python 3.14.5 successfully imports both packages
- CUDA wheel candidate confirmed in the official index: `torch 2.11.0+cu128` and `torchvision 0.26.0+cu128`, both `cp314-cp314-win_amd64`
- CUDA installation result: not completed; the large wheel transfer was interrupted before package replacement
- CUDA VERIFIED = NO

## Dependency declarations

- `requirements.txt`: does not pin a CPU-only build; it declares `torch>=2.5.0`
- `pyproject.toml`: does not declare torch or torchvision
- Future CUDA requirement, not installed now: `torch 2.11.0+cu128` with `torchvision 0.26.0+cu128` from the official `cp314-cp314-win_amd64` wheels

## Gate status

The ML environment is not CUDA-ready. Do not download models, run benchmarks, or begin the AI pipeline until the explicit CUDA wheel pair is installed and the CUDA tensor smoke test passes.

Reason CUDA VERIFIED = NO: the official CUDA wheel download was incomplete before package replacement. The current CPU PyTorch installation remains preserved.

Next blocker: network transfer of the 2.8 GB official `torch 2.11.0+cu128` wheel. No Python downgrade or random version changes are justified by the evidence.
