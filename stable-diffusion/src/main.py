import requests
import os

print("Let your mind run:")
prompt = input()

response = requests.post(
    f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
    headers={
        "authorization": f"Bearer " + str(os.environ.get("API_KEY_STABILITY")),
        "accept": "image/*"
    },
    files={"none": ''},
    data={
        "prompt": prompt,
        "output_format": "jpeg",
    },
)

if response.status_code == 200:
    with open(f"./{prompt[0:8]}.jpeg", 'wb') as file:
        file.write(response.content)
else:
    raise Exception(str(response.json()))

print(f"Image saved as {prompt[0:8]}.jpeg")