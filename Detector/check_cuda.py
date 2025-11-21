import torch

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA detected — using GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("CUDA not available — using CPU")
    
    
    
    
import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))