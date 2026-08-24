# CUDA GPU Image Processing Project

## 1. Project Overview

This project implements a GPU-accelerated image-processing pipeline using CUDA C++.

The program processes a large batch of images and performs:

1. RGB-to-grayscale conversion.
2. A 3x3 mean blur.
3. GPU execution using a CUDA kernel.
4. Multiple CUDA streams for asynchronous processing.
5. CUDA events for GPU-side timing.
6. A CPU implementation for comparison.
7. Output of processed images as PGM files.

The project is designed for the CUDA at Scale Independent Project requirement that the solution perform image or signal processing using GPU computation.

## 2. GPU Requirement

This is NOT a CPU-only implementation.

The image-processing operation is implemented in:

`rgb_to_gray_blur_kernel()`

The kernel is compiled and launched by `nvcc` and executes on the NVIDIA GPU.

CUDA streams are used so multiple image-processing pipelines can be queued asynchronously. CUDA events are used to measure operations associated with each stream.

NVIDIA documents CUDA streams as ordered sequences of commands and notes that commands in different streams may execute concurrently depending on device capabilities. Asynchronous transfers require appropriate host memory, which is why this project uses page-locked host buffers for the output transfers.

## 3. Dataset

The program can automatically generate a deterministic test dataset containing 100 images.

Default dataset:

- Number of images: 100
- Width: 1024 pixels
- Height: 768 pixels
- Format: PPM (P6)
- Channels: RGB

This is included so the project can be executed without downloading an external dataset.

You may also replace the generated images with your own PPM images. All input images must have the same dimensions.

## 4. Requirements

Linux environment with:

- NVIDIA GPU
- NVIDIA driver
- CUDA Toolkit
- `nvcc`
- GNU Make
- C++17-compatible CUDA compiler

Check CUDA:

```bash
nvidia-smi
nvcc --version
```

## 5. Build

```bash
make
```

or:

```bash
nvcc -O3 -std=c++17 main.cu -o cuda_image_processing
```

## 6. Run

The easiest method is:

```bash
./run.sh
```

This will:

1. Check `nvcc`.
2. Build the program.
3. Generate 100 images.
4. Process all 100 images on the GPU.
5. Save GPU results in `output/`.
6. Print CPU/GPU timing and correctness information.

You can also run manually:

```bash
./cuda_image_processing \
    --generate 100 \
    --width 1024 \
    --height 768 \
    --streams 4 \
    --input input \
    --output output
```

## 7. Command-Line Arguments

```text
--input <directory>
    Input PPM image directory.
    Default: input

--output <directory>
    Output directory.
    Default: output

--streams <number>
    Number of CUDA streams.
    Default: 4

--generate <number>
    Generate the specified number of test images.

--width <pixels>
    Width for generated images.
    Default: 1024

--height <pixels>
    Height for generated images.
    Default: 768

--help
    Show usage information.
```

Example:

```bash
./cuda_image_processing --generate 200 --width 1280 --height 720 --streams 4
```

## 8. CUDA Processing Pipeline

For every image:

```text
CPU input image
      |
      | cudaMemcpyAsync
      v
GPU RGB image
      |
      v
CUDA kernel
      |
      +--> RGB -> grayscale
      |
      +--> 3x3 mean blur
      |
      v
GPU processed image
      |
      | cudaMemcpyAsync
      v
CPU output buffer
      |
      v
PGM output image
```

Images are assigned to multiple CUDA streams in a round-robin manner.

For example, with four streams:

```text
Stream 0: image 0 -> GPU -> output
Stream 1: image 1 -> GPU -> output
Stream 2: image 2 -> GPU -> output
Stream 3: image 3 -> GPU -> output

Stream 0: image 4 -> GPU -> output
Stream 1: image 5 -> GPU -> output
...
```

The use of separate streams allows independent operations to be queued without forcing the entire batch into one sequential stream.

## 9. CUDA Kernel

The kernel assigns one CUDA thread to each image pixel.

Each thread:

1. Reads one RGB pixel.
2. Calculates grayscale intensity.
3. Reads the surrounding 3x3 neighborhood.
4. Calculates the mean.
5. Writes the filtered result.

The main kernel is:

```cpp
__global__ void rgb_to_gray_blur_kernel(...)
```

The launch configuration uses 16x16 CUDA threads per block.

## 10. CUDA Events

The program creates a start and stop CUDA event for each stream:

```cpp
cudaEventRecord(start_events[s], streams[s]);
...
cudaEventRecord(stop_events[s], streams[s]);
```

The elapsed time is obtained using:

```cpp
cudaEventElapsedTime(...)
```

The program also reports total batch wall time.

## 11. CPU Comparison

A CPU implementation performs the same grayscale and 3x3 blur operations.

The final output compares:

```text
CPU batch time
GPU batch wall time
CPU/GPU speedup
Average CPU/GPU pixel difference
```

The average pixel difference should normally be small because both implementations perform the same mathematical operations, although floating-point conversion and integer truncation can produce small differences.

## 12. Example Output

Your exact numbers will depend on the NVIDIA GPU and system.

Example format:

```text
CUDA GPU: NVIDIA ...
Compute capability: ...
Images found: 100
CUDA streams: 4

===== RESULTS =====
Images processed: 100
Image size: 1024x768
CPU batch time: ....... ms
GPU batch wall time: ... ms
Sum of per-stream event times: ... ms
CPU/GPU wall-time speedup: ...x
Average CPU/GPU absolute pixel difference: ...
Output directory: output
STATUS: SUCCESS
```

Do not copy these example timing values into the submission. Use the actual values produced by your execution.

## 13. Proof of Execution for Submission

For the peer-review submission, capture a screenshot showing:

- NVIDIA GPU name.
- Number of images processed.
- CUDA streams.
- CPU time.
- GPU time.
- Speedup.
- Pixel difference.
- `STATUS: SUCCESS`.

Also include several generated output images in the repository if required by your course submission rules.

The important point is to demonstrate that the program processed a large amount of data in one execution rather than only one image.

## 14. Suggested GitHub Repository

Recommended structure:

```text
CUDA-Image-Processing/
├── main.cu
├── Makefile
├── run.sh
├── README.md
├── input/
└── output/
```

Do not commit a huge dataset if your repository has size restrictions. The included program can regenerate the 100-image test dataset.

## 15. Project Description for Peer Review

This project implements a GPU-accelerated image-processing pipeline using CUDA C++. The program processes 100 RGB images and performs grayscale conversion followed by a 3x3 mean blur. The processing operation is implemented as a CUDA kernel where individual GPU threads process image pixels in parallel. Multiple CUDA streams are used to queue independent image-processing operations asynchronously, while CUDA events are used for timing. Page-locked host memory is used for asynchronous device-to-host transfers. A CPU implementation performs the same operations so that CPU and GPU execution times can be compared. The project demonstrates how GPU parallelism and asynchronous CUDA execution can be applied to a large image-processing workload.

## 16. Lessons Learned

The project demonstrates that image processing contains substantial pixel-level parallelism that can be mapped naturally to CUDA threads. It also demonstrates that simply moving a kernel to the GPU is not the entire optimization problem: host-to-device transfers, device-to-host transfers, synchronization, and workload size can influence total performance. CUDA streams provide a mechanism for organizing independent operations, while CUDA events provide GPU-side timing and synchronization.

## 17. Limitations

This implementation intentionally uses the simple PPM/PGM formats so that the project does not require OpenCV or another external image library.

For a more advanced version, the same CUDA processing kernel could be integrated with OpenCV, NVIDIA NPP, JPEG/PNG decoding, or a larger real-world image dataset.

## 18. Clean Build

```bash
make clean
```

Then:

```bash
make
```

## 19. References

NVIDIA CUDA C++ Programming Guide:
https://docs.nvidia.com/cuda/cuda-programming-guide/

NVIDIA CUDA Toolkit Documentation:
https://docs.nvidia.com/cuda/doc/
