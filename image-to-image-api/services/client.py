
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image


class Image2ImageClient:
    __MODEL = "runwayml/stable-diffusion-v1-5"
    __PIPELINE = None
    __IMAGE = None

    def __init__(self):
        self.__PIPELINE = StableDiffusionImg2ImgPipeline.from_pretrained(self.__MODEL, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
        self.__PIPELINE.unet.to(memory_format=torch.channels_last)
        self.__PIPELINE.unet = torch.compile(self.__PIPELINE.unet, mode="reduce-overhead", fullgraph=True)

    def use_gpu(self):
        self.__PIPELINE = self.__PIPELINE.to("cuda")

    def load_image(self, image_path):
        image = Image.open(image_path)
        self.__IMAGE = load_image(image)

    def generate(self, prompt):
        return self.__PIPELINE(prompt=prompt, image=self.__IMAGE).images
