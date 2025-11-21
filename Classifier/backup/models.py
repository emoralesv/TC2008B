from __future__ import annotations
import time
from typing import Sequence
import torch.nn as nn
import torch
import os
from tempfile import TemporaryDirectory
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from torchvision.transforms import functional as F
import torchvision.transforms.v2 as v2
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
### Import classifiers
from classifiers import ClassicClassifier, CascadeClassifier, GatedClassifier
from backbones import build_feature_extractor
from sklearn.metrics import f1_score



_feature_dim={
    "resnet18":512,
    "resnet34":512,
    "resnet50":2048,
    "resnet101":2048,
    "resnet152":2048
}

import torch
import torch.nn as nn

class LearnableNormalization(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        init_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        init_std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        if num_channels == 3:
            self.mean = nn.Parameter(init_mean.clone())
            self.std = nn.Parameter(init_std.clone())
        else:
            self.mean = nn.Parameter(init_mean.mean().clone().detach().unsqueeze(0))
            self.std = nn.Parameter(init_std.mean().clone().detach().unsqueeze(0))
        self.mean.requires_grad = False
        self.std.requires_grad = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.view(1, -1, 1, 1)
        std  = self.std.view(1, -1, 1, 1).clamp(min=1e-6)
        return (x - mean) / std

    
def _backbone_generator(backbone_type,channels) -> nn.Module:
    return build_feature_extractor(channels,backbone_type)
    

def _classifier_generator(fusion_type,num_views, num_classes,feature_dim)-> nn.Module:

    classifier = {
        "gated":GatedClassifier(num_views,feature_dim, num_classes), #(self,num_views, feature_dim,num_classes):
        "classic":ClassicClassifier(num_views,feature_dim, num_classes),
        "cascade":CascadeClassifier(num_views,feature_dim, num_classes),
    }
    return classifier[fusion_type]


class multiviewResnet(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        num_classes: int,
        backbone_type: str, #resnet18, resnet34, resnet50
        fusion_type: str,  #gated, bottleneckfusion, concat, 
        dynamic_gating: bool = False,
        ):
        super().__init__()
        self.dynamic_gating = dynamic_gating
        self.backbone_type = backbone_type
        self.normalizers = nn.ModuleList([LearnableNormalization(c) for c in channels])
        self.backbones = nn.ModuleList([_backbone_generator(backbone_type,c) for c in channels])
        self.classifier = _classifier_generator(fusion_type, channels.__len__(), num_classes,_feature_dim[backbone_type])
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in channels])
        if dynamic_gating:
            self.gates.requires_grad_ = True
    def unfreeze_model(self,unfreeze_gates = False):
        for param in self.parameters():
            param.requires_grad = True
        for gate in self.gates:
            gate.requires_grad = not unfreeze_gates

    def _show_trainable_layers(self):
        print(f"{'Layer':50} | Trainable")
        print("-" * 65)
        for name, param in self.named_parameters():
            print(f"{name:50} | {param.requires_grad}")
    def _show_gates(self):
        for i, gate in enumerate(self.gates):
            print(f"Gate {i}: {gate.item():.4f} (sigmoid: {torch.sigmoid(gate).item():.4f})")
    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            views = list(inputs[0])
        else:
            views = list(inputs)
        features = []
        normalized_views = [norm(xi) for norm, xi in zip(self.normalizers, views)]
        for view, backbone, gate in zip(normalized_views, self.backbones, self.gates):
            feature = backbone(view)
            if self.dynamic_gating:
                features.append(feature)
            else:
                if torch.sigmoid(gate) > 0:
                    features.append(feature)
                else:
                    features.append(torch.zeros_like(feature))
        out = self.classifier(features,self.gates)
        return out


            
            
    def train_model(self, dataloaders, dataset_sizes, criterion, optimizer, eta_min, device, warmup_epochs=5, num_epochs=25, path='./model'):
        warmup_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        cosine_scheduler = None  

        num_epochs += warmup_epochs
        since = time.time()
        train_losses = []
        val_losses = []

        with TemporaryDirectory() as tempdir:
            best_model_path = os.path.join(path, 'best_model.pt')
            best_acc = 0.0
            torch.save(self.state_dict(), best_model_path)

            for epoch in range(num_epochs):
                print(f'\nEpoch {epoch}/{num_epochs - 1}\n{"-" * 20}')
                if epoch < warmup_epochs:
                    print(f"Warming up the model ... {epoch + 1}/{warmup_epochs}")  # +1 for display
                if epoch == warmup_epochs:
                    print("Unfreezing the model for fine-tuning ...")
                    self.unfreeze_model(unfreeze_gates=self.dynamic_gating)
                    remaining = num_epochs - warmup_epochs  # equals original num_epochs
                    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=eta_min)

                for phase in ['train', 'val']:
                    self.train() if phase == 'train' else self.eval()
                    running_loss = 0.0
                    running_corrects = 0

                    all_preds = []
                    all_labels = []
                    total_batches = len(dataloaders[phase]) if dataloaders[phase] is not None else 0
                    progress_interval = max(1, total_batches // 5) if total_batches else 1
                    samples_processed = 0

                    print(f'-> Phase: {phase} | batches: {total_batches}')

                    for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                        if isinstance(inputs, (tuple, list)):
                            inputs = [i.to(device) for i in inputs]
                        else:
                            inputs = inputs.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.forward(*inputs) if isinstance(inputs, (list, tuple)) else self.forward(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        batch_size = labels.size(0)
                        running_loss += loss.item() * batch_size
                        running_corrects += torch.sum(preds == labels.data)
                        samples_processed += batch_size

                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                        if ((batch_idx + 1) % progress_interval == 0) or (batch_idx + 1 == total_batches):
                            avg_loss = running_loss / max(samples_processed, 1)
                            acc_value = (running_corrects.double() / max(samples_processed, 1)).item()
                            print(f'   [{phase}] batch {batch_idx + 1}/{total_batches} | loss: {avg_loss:.4f} | acc: {acc_value:.4f}')

                    # ---- per-epoch metrics ----
                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()  # ✅ make float
                    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

                    if phase == 'train':
                        train_losses.append(epoch_loss)
                    else:
                        val_losses.append(epoch_loss)

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(self.state_dict(), best_model_path)

                # ---- step the correct scheduler once per epoch (AFTER optimizer updates) ----
                if epoch < warmup_epochs:
                    warmup_scheduler.step()
                else:
                    if cosine_scheduler is not None:  # ✅ guard
                        cosine_scheduler.step()

            total_time = time.time() - since
            print(f'\nTraining complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
            print(f'Best val Acc: {best_acc:.4f}')
            self.load_state_dict(torch.load(best_model_path))  # ✅ no weights_only

        return self, train_losses, val_losses, best_model_path
    

             

if __name__ == "__main__":
    # --- configuration ---
    channels = [3, 1]
    num_classes = 2
    backbone_type = "resnet18"
    fusion_type = "cascade"  # Options: "gated", "classic", "cascade"
    dynamic_gating = False  # Set to True to enable dynamic gating
    print("\n" + "=" * 80)
    print("🚀 Initializing Multi-View ResNet")
    print("=" * 80)
    print(f"Backbone Type : {backbone_type}")
    print(f"Fusion Type   : {fusion_type}")
    print(f"Views (channels): {channels}")
    print(f"Num Classes   : {num_classes}")
    print("-" * 80)

    # --- build model ---
    model = multiviewResnet(
        channels=channels,
        num_classes=num_classes,
        backbone_type=backbone_type,
        fusion_type=fusion_type,
    )

    # --- summary ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model Summary")
    print("-" * 80)
    print(model)
    print("-" * 80)
    print(f"Total parameters   : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 80)

    # --- trainable layer overview ---
    print("\n🔍 Trainable Layers\n" + "-" * 80)
    model._show_trainable_layers()
    model._show_gates()
    # --- optional dummy test ---
    print("\n🧪 Running dummy forward pass ...")
    B, H, W = 3, 224, 224
    dummy_views = [torch.randn(B, c, H, W) for c in channels]
    with torch.no_grad():
        out = model(dummy_views)
    print(out)
    print(f"✅ Output shape: {tuple(out.shape)}  (expected: [{B}, {num_classes}])")
    print("=" * 80 + "\n")
    
    
    
