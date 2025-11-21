from ultralytics import YOLO, settings
import matplotlib.pyplot as plt
import torch



if torch.cuda.is_available():
    device = "cuda"
    print("CUDA available — using GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("CUDA not available — using CPU (auto-batch may be slow)")
    
    

DATA_YAML = "Rugoso_Test-1/data.yaml"  
N_EPOCHS = 180              
PROJECT_NAME = "rugoso"

MODEL_SIZES = ["m","l","x"]


results_summary = []


for size in MODEL_SIZES:
    model_name = f"yolo11{size}.pt"
    exp_name = f"exp_{size}"

    print(f"\n==============================")
    print(f"Training {model_name} for {N_EPOCHS} epochs")
    print(f"Project: {PROJECT_NAME}, Name: {exp_name}")
    print(f"==============================\n")

    # Load model
    model = YOLO(model_name)

    # Train
    train_results = model.train(
        data=DATA_YAML,
        epochs=N_EPOCHS,
        imgsz=720,
        batch=4,         # auto batch size
        degrees = 10,
        flipud=0.5,
        shear = 1,
        perspective = 0.0003,
        project=PROJECT_NAME,
        name=exp_name,
        device=device,
        erasing = 0.15,
    )

    # Validate on the same dataset 
    val_results = model.val(data=DATA_YAML)

    # mAP (mAP@0.5:0.95)
    map_5095 = val_results.box.map

    # Inference speed per image in ms
    # val_results.speed is a dict like:
    # {'preprocess': x, 'inference': y, 'postprocess': z}
    inf_ms = val_results.speed['inference']

    print(f"Validation results for {model_name}:")
    print(f"  mAP@0.5:0.95 = {map_5095:.4f}")
    print(f"  Inference time (ms/img) = {inf_ms:.2f}")

    # Save for later plotting
    results_summary.append({
        "size": size,
        "model": model_name,
        "map": map_5095,
        "inference_ms": inf_ms,
    })



print("\n===== SUMMARY =====")
print(f"{'Model':<12} {'Size':<5} {'mAP@0.5:0.95':<15} {'Inf ms/img':<10}")
for r in results_summary:
    print(f"{r['model']:<12} {r['size']:<5} {r['map']:<15.4f} {r['inference_ms']:<10.2f}")



maps = [r["map"] for r in results_summary]
infs = [r["inference_ms"] for r in results_summary]
labels = [r["size"] for r in results_summary]

plt.figure()
plt.scatter(infs, maps)

for x, y, label in zip(infs, maps, labels):
    plt.text(x, y, label, fontsize=9, ha='right', va='bottom')

plt.xlabel("Inference time (ms / image)")
plt.ylabel("mAP@0.5:0.95")
plt.title("YOLOv8 variants: Inference time vs mAP (Rugoso)")
plt.grid(True)
plt.tight_layout()
plt.show()
