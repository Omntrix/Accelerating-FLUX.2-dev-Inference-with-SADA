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

image.save("flux2_output.png")