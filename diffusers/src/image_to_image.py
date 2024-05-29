import torch
from diffusers import AutoPipelineForImage2Image, DPMSolverMultistepScheduler
from diffusers.utils import load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
# Move the pipeline to the GPU for faster processing (optional)
# pipeline = pipeline.to("cuda")
#pipeline.enable_model_cpu_offload()
#pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
#my lkdn profile pic
init_image = load_image("https://media.licdn.com/dms/image/D4E03AQEs5nPKFAZnnA/profile-displayphoto-shrink_200_200/0/1692742003003?e=2147483647&v=beta&t=L3BR84t-eVZ6lA9SltljWD-IfYKcXJAN2GWYAR9i-3g")

prompt = "Anime, Dragon Ball Style, High Resolution, 8k"
image = pipeline(prompt, image=init_image).images[0]
image.save(f"./me_dbz.jpeg")