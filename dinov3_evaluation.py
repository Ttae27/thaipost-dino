import os
import time
import random
import argparse

import psutil
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from datasets import CargoSpaceDataset, PLOT_CLASS_NAMES
from models import Network


def save_visualizations(image_records, test_dataset, vis_dir, num_classes=6, max_correct_per_class=20):
    correct_root = os.path.join(vis_dir, "correct")
    false_root = os.path.join(vis_dir, "false")

    selected_records = []
    for c in range(num_classes):
        class_imgs = [r for r in image_records if r['true_label'] == c]
        correct_imgs = [r for r in class_imgs if r['pred_label'] == r['true_label']]
        false_imgs = [r for r in class_imgs if r['pred_label'] != r['true_label']]

        sampled_correct = random.sample(correct_imgs, min(max_correct_per_class, len(correct_imgs)))
        selected_records.extend(sampled_correct)
        selected_records.extend(false_imgs)

    for record in selected_records:
        img_path, _ = test_dataset.samples[record['idx']]
        original_image = Image.open(img_path).convert("RGB")

        pred_text = PLOT_CLASS_NAMES[record['pred_label']]
        true_text = PLOT_CLASS_NAMES[record['true_label']]
        conf_score = record['confidence']
        is_correct = record['pred_label'] == record['true_label']

        plt.figure(figsize=(6, 6))
        plt.imshow(original_image)
        plt.axis('off')

        title_color = 'green' if is_correct else 'red'
        plt.title(
            f"Predict: {pred_text} ({conf_score:.1f}%)\nGround Truth: {true_text}",
            color=title_color, fontsize=14, fontweight='bold', pad=10,
        )

        class_num = true_text.split('(')[-1].strip(')')
        bucket_root = correct_root if is_correct else false_root
        save_folder = os.path.join(bucket_root, f"class {class_num}")
        os.makedirs(save_folder, exist_ok=True)

        clean_pred = pred_text.replace(" ", "").replace("/", "-")
        clean_true = true_text.replace(" ", "").replace("/", "-")
        filename = f"img_{record['idx']:04d}_T[{clean_true}]_P[{clean_pred}].png"

        plt.savefig(os.path.join(save_folder, filename), bbox_inches='tight', dpi=100)
        plt.close()

    return len(selected_records)


def evaluate_model(model, test_loader, test_dataset, device, args, num_classes=6):
    model.eval()

    print("\n[1/4] Measuring Accuracy, Precision, Recall & F1...")
    val_total = 0
    val_correct = 0

    all_preds = []
    all_labels = []
    image_records = []
    sample_idx = 0

    with torch.no_grad():
        for val_images, val_labels in test_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            out = model(val_images)
            probabilities = F.softmax(out, dim=1)
            confidences, val_predicted = torch.max(probabilities, 1)

            val_total += val_labels.size(0)
            val_correct += val_predicted.eq(val_labels).sum().item()

            all_preds.extend(val_predicted.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())

            if args.save_vis:
                for i in range(val_images.size(0)):
                    image_records.append({
                        'idx': sample_idx,
                        'true_label': val_labels[i].item(),
                        'pred_label': val_predicted[i].item(),
                        'confidence': confidences[i].item() * 100,
                    })
                    sample_idx += 1

    overall_accuracy = val_correct / val_total if val_total > 0 else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0,
    )

    class_report = classification_report(
        all_labels, all_preds, target_names=PLOT_CLASS_NAMES, zero_division=0,
    )

    print("[2/4] Analyzing Hardware & Specs...")
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 ** 2)
    peak_ram_mb = process.memory_info().peak_wset / (1024 ** 2) if os.name == 'nt' else None

    print("[3/4] Running Latency & Concurrency Benchmark...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    for _ in range(10):
        _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    iterations = 100
    latencies = []

    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.time()
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.time() - start_time)

    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    fps = 1000 / avg_latency_ms

    vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0

    print("[4/4] Generating Confusion Matrix Plot...")
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=PLOT_CLASS_NAMES, yticklabels=PLOT_CLASS_NAMES)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Cargo Space Confusion Matrix\nAccuracy: {overall_accuracy*100:.2f}%')

    safe_weights_name = os.path.basename(args.weights).replace('.pth', '')
    cm_filename = f"confusion_matrix_{safe_weights_name}.png"
    plt.savefig(cm_filename, bbox_inches='tight')
    plt.close()

    saved_vis_count = 0
    if args.save_vis:
        print(f"[+] Saving Visualization Images to ./{args.vis_dir}/ ...")
        saved_vis_count = save_visualizations(
            image_records, test_dataset, args.vis_dir,
            num_classes=num_classes, max_correct_per_class=args.vis_max_correct,
        )

    print("\n" + "=" * 50)
    print("                 EVALUATION REPORT")
    print("=" * 50)
    print(f"Dataset Size:      {val_total} images")
    print(f"Overall Accuracy:  {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1-Score:    {f1:.4f}")
    print("-" * 50)
    print("             PER-CLASS BREAKDOWN")
    print("-" * 50)
    print(class_report)
    print("-" * 50)
    print("               SPECS CONSUMPTION")
    print("-" * 50)
    print(f"System RAM Usage:  {ram_mb:.2f} MB")
    print(f"GPU VRAM Peak:     {vram_mb:.2f} MB")
    print("-" * 50)
    print("           LATENCY & CONCURRENCY (BS=1)")
    print("-" * 50)
    print(f"Avg Latency:       {avg_latency_ms:.2f} ms / image")
    print(f"Throughput (FPS):  {fps:.2f} Frames Per Second")
    print("=" * 50)
    print(f"--> Confusion Matrix saved as: {cm_filename}")
    if args.save_vis:
        print(f"--> Saved {saved_vis_count} visualization images to: ./{args.vis_dir}/")
    print("=" * 50)


def get_args():
    parser = argparse.ArgumentParser(description="Cargo Space Evaluation Script")
    parser.add_argument('-w', '--weights', type=str, required=True, help="Path to the .pth weights file")
    parser.add_argument('-d', '--dataset', type=int, choices=[0, 1], required=True, help="Dataset split logic")
    parser.add_argument('--data_dir', type=str, default='Cargo space', help="Root directory of the dataset")
    parser.add_argument('-a', '--attention', type=str, choices=['none', 'dam'], required=True, help="Attention mechanism")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for accuracy test")
    parser.add_argument('--save_vis', action='store_true',
                        help="Save per-image visualization (prediction vs ground truth)")
    parser.add_argument('--vis_dir', type=str, default='predict',
                        help="Directory to save visualization images (used with --save_vis)")
    parser.add_argument('--vis_max_correct', type=int, default=20,
                        help="Max correct samples to save per class. All wrong predictions are always saved. Set to 0 to save only wrong predictions.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device.type.upper()}")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = CargoSpaceDataset(root_dir=args.data_dir, dataset_type=args.dataset, mode='test', transform=transform)

    if len(test_dataset) == 0:
        print(f"Error: No images found under '{args.data_dir}'. Check the path and class folder names.")
        exit(1)

    print(f"Total testing images: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device_map = "cuda" if device.type == "cuda" else "cpu"
    model = Network(backbone=None, attention=args.attention, class_dim=6, device_map=device_map).to(device)

    try:
        model.load_state_dict(torch.load("checkpoints/" + args.weights, map_location=device))
        print(f"Successfully loaded weights from: {args.weights}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit(1)

    evaluate_model(model, test_loader, test_dataset, device, args)
