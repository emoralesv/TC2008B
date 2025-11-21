
from torchvision.models import resnet18,resnet34,resnet50,resnet101,resnet152
from torchvision.models import ResNet18_Weights,ResNet34_Weights,ResNet50_Weights,ResNet101_Weights,ResNet152_Weights
from torch import nn

weights_init = {
"resnet18":ResNet18_Weights.DEFAULT,
"resnet34":ResNet34_Weights.DEFAULT,
"resnet50":ResNet50_Weights.DEFAULT,
"resnet101":ResNet101_Weights.DEFAULT,
"resnet152":ResNet152_Weights.DEFAULT
}
models_init ={
"resnet18":resnet18,
"resnet34":resnet34,
"resnet50":resnet50,
"resnet101":resnet101,
"resnet152":resnet152
}


def build_feature_extractor(in_channels,model):
    if model not in models_init:
        raise ValueError(f"Model {model} not recognized. Available models: {list(models_init.keys())}")
    if in_channels == 3:
        model = _freeze_model(models_init[model](weights=weights_init[model]))
        model.need_warmup = False 
    else:
        model = _freeze_model(models_init[model](weights=weights_init[model]))
        model = _generate_input_layer(in_channels, model)
        model.need_warmup = True
    model.fc = nn.Identity()
    
    
    return model

def _freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def _generate_input_layer(in_channels, model):
    old_conv = model.conv1
    out_channels = old_conv.out_channels
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride
    padding = old_conv.padding
    new_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
    model.conv1 = new_conv
    return model
def _show_trainable_layers(model):
    print(f"{'Layer':50} | Trainable")
    print("-" * 65)
    for name, param in model.named_parameters():
        print(f"{name:50} | {param.requires_grad}")


if __name__ == "__main__":
    model = build_feature_extractor(3,"resnet18")
    print(model)
    _show_trainable_layers(model)


