import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import argparse
import yaml
import cv2

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_dir, ".."))

from engine.training_utils import setup_device, select_model, load_config

def load_transforms(config):
    return transforms.Compose([
        transforms.Resize((config["IMAGE_SIZE"], config["IMAGE_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=eval(str(config["IMAGENET_MEAN"])),
            std=eval(str(config["IMAGENET_STD"])),
        ),
    ])

def load_model(config, checkpoint_path, device):
    model = select_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def predict_image(model, transform, image_path, device, class_names):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class

def main(args):
    config = load_config(args.config_path)
    device = setup_device()
    transform = load_transforms(config)
    class_names = [str(i) for i in range(config["NUM_CLASSES"])]  # O usa get_class_names()

    model = load_model(config, args.checkpoint_path, device)

    if os.path.isdir(args.input_path):
        image_files = [f for f in os.listdir(args.input_path) if f.lower().endswith((".jpg", ".png"))]
        for fname in image_files:
            path = os.path.join(args.input_path, fname)
            pred = predict_image(model, transform, path, device, class_names)
            print(f"{fname}: {pred}")
            
    elif args.input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(args.input_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Guardar frame temporalmente
            temp_path = f"temp_frame_{frame_idx}.jpg"
            cv2.imwrite(temp_path, frame)
            pred = predict_image(model, transform, temp_path, device, class_names)
            print(f"Frame {frame_idx}: {pred}")
            os.remove(temp_path)
            frame_idx += 1
        cap.release()
        
    else:
        pred = predict_image(model, transform, args.input_path, device, class_names)
        print(f"{args.input_path}: {pred}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Ruta al archivo config.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Ruta al modelo entrenado .pt")
    parser.add_argument("--input_path", type=str, required=True, help="Imagen o carpeta de imágenes para inferencia")
    args = parser.parse_args()
    main(args)
