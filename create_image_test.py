import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os


if not load_dotenv():
    print("WARNING: .env file not found or cannot be loaded.")

client = InferenceClient(
    provider="nebius",
    api_key=os.environ["HF_TOKEN"],
)

prompt = "A beautiful ancient-style maiden in soft flowing hanfu, semi-realistic style, long black hair, gentle expression, front view clear face, warm golden backlight, birds flying around, dreamy atmosphere, soft lighting, elegant composition."
st_time = time.time()
image = client.text_to_image(
    prompt,
    model="black-forest-labs/FLUX.1-schnell",
)
end_time = time.time()

# convert về RGB để lưu đuôi .png
png_image = image.convert("RGB")
png_image.save("output1.png", format="PNG")
print("Saved output.png")
print(f"Time: {end_time - st_time}")