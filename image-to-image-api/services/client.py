
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from PIL import Image


class Image2ImageClient:
    __MODEL = "runwayml/stable-diffusion-v1-5"
    __PIPELINE = None
    __IMAGE = None

    def __init__(self):
        self.__PIPELINE = AutoPipelineForImage2Image.from_pretrained(self.__MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

    def use_gpu(self):
        self.__PIPELINE = self.__PIPELINE.to("cuda")

    def load_image(self, image_path):
        image = Image.open(image_path)
        self.__IMAGE = load_image(image)

    def generate(self, prompt):
        return self.__PIPELINE(prompt, image=self.__IMAGE).images
