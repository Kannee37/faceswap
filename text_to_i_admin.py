from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os
import argparse

if not load_dotenv():
    print("WARNING: .env file not found or cannot be loaded.")

client = InferenceClient(
    provider="nebius",
    api_key=os.environ["HF_TOKEN"],
)

def text_to_image(base_prompt:str, output_path: str):
   image = client.text_to_image(
      base_prompt,
      model="black-forest-labs/FLUX.1-schnell",
   )
   image.save(output_path)
   return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text to Image")
    parser.add_argument("--prompt", type=str, required=True,
                        help="base prompt tạo ảnh")
    parser.add_argument("--output", type=str, required=True,
                        help="Đường dẫn output")
    args = parser.parse_args()
    text_to_image(args.prompt, args.output)