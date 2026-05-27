# Nerfstudio Project Setup

## CUDA 11.8 Setup (Windows)

This project was tested with:

* CUDA 11.8
* PyTorch CUDA 11.8 build
* Visual Studio 2022 Build Tools
* MSVC 14.38
* Python 3.10
* Windows SDK 10.0.26100.0

### Required environment setup

Before installing `tiny-cuda-nn` or compiling CUDA extensions, the following environment variables were required:

```bat
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

set PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64;%PATH%

set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH%

set LIB=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38.33130\lib\x64;%LIB%

set INCLUDE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38.33130\include;%INCLUDE%

set TCNN_CUDA_ARCHITECTURES=89
```

### tiny-cuda-nn installation

```bat
pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Nerfstudio installation

```bat
pip install nerfstudio
```

### Important Notes

* `cl.exe`, `link.exe`, `rc.exe`, and `mt.exe` must all be accessible from the terminal.
* Missing `LIB` or `INCLUDE` paths can cause errors such as:

  * `MSVCRTD.lib not found`
  * `yvals.h not found`
  * `crtdefs.h not found`
* CUDA 11.8 worked correctly with this setup.
