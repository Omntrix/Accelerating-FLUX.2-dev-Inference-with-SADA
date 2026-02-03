# Accelerating-FLUX.2-dev-Inference-with-SADA
Accelerate FLUX.2 inference from 18s to 12s (33% speedup) using SADA in H200
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python-Version](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](https://www.python.org/)

This repository implements the **SADA: Stability-guided Adaptive Diffusion Acceleration** algorithm for the **FLUX.2-dev** model.

## 🚀 Performance Benchmarks

In our tests, Flux2-SADA reduces inference latency by **33%**.

| Implementation | Latency (Per Image) | Speedup | GPU | Resolution | Preview |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FLUX.2 (dev)** | 18.0s | 1.0x | H200 | 1024x1024 | ![Baseline](assets/baseline.png) |
| **FLUX.2 + SADA** | **12.0s** | **1.5x (🚀)** | H200 | 1024x1024 | ![SADA](assets/sada_result.png) |

*Note: Benchmarks were conducted with 30 inference steps and BF16 precision. Results may vary depending on hardware and parameters.*

## 🛠️ Key Features
- **33% Speedup**: Optimized for FLUX.2's transformer architecture.
- **Plug-and-Play**: Easily integrates with the HuggingFace `diffusers` library.
- **Quality Preserving**: Maintains high-fidelity structure and details while skipping redundant computations.

## 📦 Installation
```bash
- python3.12 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
```
## ⚡Quickstart
```bash
import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from sada import patch
repo_id = "/data/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/..."
device = "cuda:0"
torch_dtype = torch.bfloat16

pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype
)
pipe = pipe.to(device)

patch.apply_patch(
                pipe,
                max_downsample=0,
                acc_range=(8, 27),
                latent_size=(1024 // 16, 1024 // 16), 
                lagrange_int=4,
                lagrange_step=20,
                lagrange_term=3,
                max_fix=0,
                max_interval=4
            )

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

#cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")

image = pipe(
    prompt=prompt,
    #image=[cat_image] #multi-image input
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=30,
    guidance_scale=4,
).images[0]

image.save("flux2_output1.png")
```

## 🤝 Acknowledgements
Special thanks to [Ting-Justin-Jiang/sada-icml](https://github.com/Ting-Justin-Jiang/sada-icml) for their outstanding algorithmic contributions.

## 📜 License
This project adheres to the [Black Forest Labs FLUX.1/2 Model License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX.1-dev).
