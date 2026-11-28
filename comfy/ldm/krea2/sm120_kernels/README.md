# Krea 2 Turbo SM120 profile

This optional profile targets Windows x64, GeForce RTX 5090, CUDA 13 PyTorch, and Krea 2 Turbo FP8 scaled. It supports single-image sampling at variable resolutions and prompt lengths with FP16 compute and accumulation, eight Euler/simple steps, CFG 1, and denoise 1. Unsupported workflows use stock sampling.

## Install and enable

Run these commands from the ComfyUI directory with the Python executable used to launch ComfyUI.

```powershell
python -m pip install --only-binary=:all: -r requirements-krea2-sm120.txt
python main.py --fp16-unet --use-pytorch-cross-attention --fast fp16_accumulation --krea2-sm120
```

The optional requirements install a prebuilt adapter wheel and NVIDIA's cuBLAS package from PyPI. Allow approximately 475 MB of downloads for a first install. The profile also works with `--use-sage-attention` in an environment with SageAttention already installed. The profile keeps the selected attention backend.

After warming the model, an eligible render logs its MXFP8, FP16, normalization, modulation, and text-preparation counts. You can add `--fast-unfiltered-rgb-png` to write lossless uncompressed RGB `SaveImage` output with text metadata. These files are larger than compressed PNGs.

## Build from source

To rebuild the adapters, use CUDA Toolkit 13.0 and the MSVC 2022 C++ x64 tools. Set `CUDA_PATH` to the toolkit or supply `-CudaRoot`. The build verifies the SHA-256 of NVIDIA's cuBLASLt 13.8.0.4 DLL. [CUDA redistributables manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.3.json)

```powershell
& .\comfy\ldm\krea2\sm120_kernels\build.ps1 -CublasLt C:\cuda-sdk\cublasLt64_13.dll -OutputDirectory C:\krea2-sm120
python -m pip install wheel
python comfy/ldm/krea2/sm120_kernels/pack_wheel.py C:\krea2-sm120 dist
```

For a local build, pass the output directory directly as `--krea2-sm120 C:\krea2-sm120`. The wheel contains the four compiled adapters and their source.

Projection shapes with 16,862 rows can use tuned cuBLASLt algorithms. Other shapes can use the generic MXFP8 path linked against the PyTorch cuBLASLt DLL. The FP8 projections retain the original E4M3 codes and FP32 scales. The CUDA modulation kernel rounds the multiply to FP16 before addition.
