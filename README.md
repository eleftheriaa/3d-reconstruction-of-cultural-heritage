# NeRF Environment Setup (Windows)

## Requirements

- Python 3.10
- CUDA 11.8
- Visual Studio 2022 Build Tools
  - MSVC v143 build tools
  - Windows 11 SDK
- CMake
- Ninja

---

## Create Conda Environment

```cmd
conda create -n nerfstudio python=3.10 -y
conda activate nerfstudio
```

---

## Install PyTorch (CUDA 11.8)

```cmd
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

---

## Install Nerfstudio

```cmd
pip install nerfstudio
```

---

## Install Compatible setuptools and wheel

```cmd
pip install setuptools==69.5.1 wheel==0.43.0
```

---

## Environment Variables

```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

set PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64;%PATH%

set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH%

set LIB=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38.33130\lib\x64;%LIB%

set INCLUDE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38.33130\include;%INCLUDE%
```

---

## Install tiny-cuda-nn

```cmd
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
```

---

## Verify Compiler

```cmd
where cl
where link
where rc
where mt
```