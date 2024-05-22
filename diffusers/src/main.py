import torch
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

# Move the pipeline to the GPU for faster processing (optional)
pipe = pipe.to("cuda")

print("Think and I'll paint it for you:")
prompt = input()

# Generate images based on the prompt
images = pipe(prompt)

# Access the generated image
generated_image = images.images[0]

generated_image.save(f"./{prompt[0:8]}.jpeg")
