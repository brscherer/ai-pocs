import pyautogui
import base64
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO

load_dotenv()

client = OpenAI()

screenshot = pyautogui.screenshot()
buffer = BytesIO()
screenshot.save(buffer, format="PNG")
buffer.seek(0)
img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{img_base64}",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])