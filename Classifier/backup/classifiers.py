from torch import nn
import torch
import math
class ClassicClassifier(nn.Module):
    def __init__(self,num_views,feature_dim,num_classes, gated = False):
        super().__init__()
        self.gated = gated
        in_dim = feature_dim if gated else feature_dim * num_views
        self.fc1 = nn.Linear(in_dim, in_dim//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_dim//2, num_classes)

    def forward(self, features,gates):
        if not self.gated:
            features = torch.cat(features,dim=1)
        f = self.fc1(features)
        f = self.relu(f)
        f = self.dropout(f)
        f = self.fc2(f)
        return f

    
class CascadeClassifier(nn.Module):
    def __init__(self,num_views, feature_dim,num_classes,gated = False):
        super().__init__()
        self.reductors = nn.ModuleList([self._reductor(feature_dim, num_views) for _ in range(num_views)])
        self.classifier = ClassicClassifier(num_views,feature_dim//num_views,num_classes,gated)
    def _reductor(self, dim, num_views):
        layers = []
        target_dim = dim // num_views
        while dim // 2 >= target_dim:
            layers.append(nn.Linear(dim, dim // 2))
            layers.append(nn.ReLU())
            dim = dim // 2
        layers.append(nn.Dropout(0))
        return nn.Sequential(*layers)
    def _setDropout(self,p_values):
        for p,reductor in zip(p_values, self.reductors):
            for module in reductor:
                if isinstance(module,nn.Dropout):
                    module.p = p
    def forward(self,features,gates):

        if self.training and gates is not None:
            self._setDropout([1 / (1 + math.exp(-g.item())) for g in gates])
        reduced = [red(x)  for x, red, gate in zip(features, self.reductors, gates)]
        return self.classifier(reduced,gates)
        
    
class GatedClassifier(nn.Module):
    def __init__(self,num_views, feature_dim,num_classes):
        super().__init__()
        self.feature_dim = feature_dim
        self.classifier = ClassicClassifier(num_views,feature_dim,num_classes, gated = True)
    def forward(self, features, gates):
        fused = torch.zeros((self.feature_dim), dtype=features[0].dtype, device=features[0].device)
        for i, (feat, gate) in enumerate(zip(features, gates)):
            fused = fused + feat * torch.sigmoid(gate)
        return self.classifier(fused,gates)
    

if __name__ == "__main__":
    num_views = 2
    feature_dim = 2048
    num_classes = 2

    print("\n" + "="*70)
    print(f"Testing Classifiers (views={num_views}, feature_dim={feature_dim}, classes={num_classes})")
    print("="*70 + "\n")

    classifiers = {
        "ClassicClassifier": ClassicClassifier(num_views, feature_dim, num_classes),
        "GatedClassifier":   GatedClassifier(num_views, feature_dim, num_classes),
        "CascadeClassifier": CascadeClassifier(num_views, feature_dim, num_classes)
    }

    for name, clf in classifiers.items():
        print(f"\n🧠 {name}")
        print("-"*len(name)*2)
        print(clf)
        total_params = sum(p.numel() for p in clf.parameters())
        trainable_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")
        print("="*70)

