<div align="center">

# RENVI - Real-ESRGAN ncnn Vulkan Improved

![download](https://img.shields.io/github/downloads/ONdraid/Real-ESRGAN-ncnn-vulkan-improved/total)
![support](https://img.shields.io/badge/Support-Linux%20x64-blue?logo=Linux)
![license](https://img.shields.io/github/license/ONdraid/Real-ESRGAN-ncnn-vulkan-improved)

</div>

<div align="justify">

RENVI (or Real-ESRGAN ncnn Vulkan Improved) is sligthly improved version of [Real-ESRGAN ncnn Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) with additional features like piping and fast PNG output.

</div>

## Additional Features

- **Improved performance**: Uses libpng with low compression level for faster PNG writing.
- **Piping**: Supports reading from stdin and writing to stdout, allowing easy integration into pipelines.

> [!IMPORTANT]  
> Only PNG format supports piping currently. JPEG and WebP formats are not supported for stdin/stdout operations. See the example usage below.

## Installation

You can download the latest release from the [Releases page](https://github.com/ONdraid/Real-ESRGAN-ncnn-vulkan-improved/releases) or build it from source:

<details>
<summary>Dependencies</summary>

> libpng, ... (TODO: add rest of dependencies)

</details>

```shell
git clone --recurse-submodules -j8 git@github.com:ONdraid/Real-ESRGAN-ncnn-vulkan-improved.git

cd Real-ESRGAN-ncnn-vulkan-improved

mkdir build && cd build

cmake ../src

cmake --build . -j 4
```

## Usages

### Example Usage

You can create a full ffmpeg pipeline to upscale a video!

```shell
ffmpeg \
-hide_banner -loglevel error \
-i input.mkv -f image2pipe -vcodec png - \
| ./realesrgan-ncnn-vulkan \
-n realesr-animevideov3 -s 2 \
| ffmpeg \
-f image2pipe -vcodec png -i - output.mkv
```

> [!TIP]
> Ffmpeg defaults to 25 fps, use `-r` option to set the desired frame rate for the output video.

### Full usage

```
Usage: realesrgan-ncnn-vulkan [options]...

  -h                   show this help
  -i input-path        input image path (jpg/png/webp) or directory (reads from stdin if not provided)
  -o output-path       output image path (jpg/png/webp) or directory (outputs to stdout if not provided)
  -s scale             upscale ratio (can be 2, 3, 4. default=4)
  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu
  -m model-path        folder path to the pre-trained models. default=models
  -n model-name        model name (default=realesr-animevideov3, can be realesr-animevideov3 | realesrgan-x4plus | realesrgan-x4plus-anime | realesrnet-x4plus)
  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu
  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu
  -x                   enable tta mode
  -f format            output image format (jpg/png/webp, default=ext/png)
  -v                   verbose output
```

> [!NOTE]  
> When utilizing stdin/stdout, load/save threads are fixed at 1:1, only proc threads can be adjusted in this configuration.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ONdraid/Real-ESRGAN-ncnn-vulkan-improved&type=Date)](https://www.star-history.com/#ONdraid/Real-ESRGAN-ncnn-vulkan-improved&Date)
