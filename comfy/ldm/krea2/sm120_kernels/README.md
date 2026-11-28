# Krea 2 Turbo SM120 profile

`--krea2-sm120` enables an experimental Windows x64 profile for RTX 5090. It targets Krea 2 Turbo FP8 scaled with FP16 compute, Euler/simple at eight steps, CFG 1, denoise 1, and a batch of one image. The profile accepts variable image sizes and encoded prompt lengths and keeps the stock SageAttention path. Model loads and unsupported workflows use stock sampling. The selector is independent of the default `--fast` presets.

## Build

Use CUDA Toolkit 13.0, the MSVC 2022 C++ x64 tools, and a ComfyUI environment with CUDA PyTorch, SageAttention, and compatible Triton. Obtain the x86-64 cuBLAS 13.8.0.4 redistributable archive from NVIDIA and extract `cublasLt64_13.dll`. The build and runtime both check this DLL's SHA-256. [CUDA redistributables manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.3.json)

```powershell
& .\comfy\ldm\krea2\sm120_kernels\build.ps1 -CublasLt C:\cuda-sdk\cublasLt64_13.dll -OutputDirectory C:\krea2-sm120
python main.py --fp16-unet --use-sage-attention --fast fp16_accumulation --krea2-sm120 C:\krea2-sm120 --fast-unfiltered-rgb-png
```

Set `CUDA_PATH` to Toolkit 13.0 or supply `-CudaRoot` to the build. Projection shapes with 16,862 rows can use tuned cuBLASLt algorithms. Other shapes can use the generic MXFP8 path linked against the PyTorch cuBLASLt DLL. Linear operations outside these paths use the original implementation. The FP8 projections retain the original weight and activation codes. Unit E8M0 scales enable the block-scaled kernels while the original FP32 scales remain in the alpha multiplier. Modulation retains the separate FP16 rounding of the multiply and add.

After warming the model, an eligible render logs its MXFP8, FP16, normalization, modulation, and text-preparation counts. The first use of a shape can compile a Triton kernel. The optional PNG switch writes lossless uncompressed RGB `SaveImage` outputs with text metadata. These files are larger than compressed PNGs.
