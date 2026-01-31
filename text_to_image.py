from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse


if not load_dotenv():
    print("WARNING: .env file not found or cannot be loaded.")

client = InferenceClient(
    provider="nebius",
    api_key=os.environ["HF_TOKEN"],
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model resnet18_gender
class GenderResNet50(nn.Module):
  def __init__(self, num_classes = 2, pretrained = True):
    super(GenderResNet50, self).__init__()
    #load resnet18 gốc
    self.model = models.resnet18(pretrained = pretrained)
    in_features = self.model.fc.in_features
    self.model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

  def forward(self, x):
    return self.model(x)

# Hàm transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
model_path = "resnet50_gender.pth"
model = GenderResNet50(num_classes = 2, pretrained = True)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Hàm dự đoán
def predict_gender(img_path, model=model, transform=transform, device=device):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()
    return "Male" if pred == 0 else "Female"


def build_prompt_with_gender(base_prompt: str, gender: str) -> str:
   if gender == "Male":
      return f"upper body portrait, half-body shot, from waist up, a male person, {base_prompt}"
   elif gender == "Female":
      return f"upper body portrait, half-body shot, from waist up, a female person, {base_prompt}"
   else:
      return base_prompt
   

def text_to_image(img_path: str, base_prompt:str, output_path: str):
   gender = predict_gender(img_path)
   final_prompt = build_prompt_with_gender(base_prompt, gender)
   image = client.text_to_image(
      final_prompt,
      model="black-forest-labs/FLUX.1-schnell",
   )
   image.save(output_path)
   return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text to Image")
    parser.add_argument("--image", type=str, required=True,
                        help="Đường dẫn ảnh nguồn (ảnh cung cấp khuôn mặt)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="base prompt tạo ảnh")
    parser.add_argument("--output", type=str, required=True,
                        help="Đường dẫn output")
    args = parser.parse_args()
    text_to_image(args.image, args.prompt, args.output)