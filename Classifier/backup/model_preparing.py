       


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

from models import multiviewResnet

def export_onnx(model, onnx_path):
    onnx_path = Path(onnx_path)

    if onnx_path.exists():
        print(f"[INFO] ONNX model already exists → {onnx_path}")
        return

    dummy = torch.randn(1, 3, 128, 128).to(device)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )

    print(f"[INFO] Exported ONNX model → {onnx_path}")
    
    
    
num_classes = 6
channels = [3]    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
print(device)
model = multiviewResnet(
    channels=channels,
    num_classes=num_classes,
    backbone_type="resnet50",
    fusion_type="gated",
    dynamic_gating=False,
).to(device)

THIS_DIR = Path(__file__).resolve().parent 
PROJECT_ROOT = THIS_DIR  
MODEL_PATH = THIS_DIR / "best_model.pt"
weights = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(weights)
model.to(device)
model.eval() 
print("Model loaded and set to eval mode.")


export_onnx(model, "best_model.onnx")


