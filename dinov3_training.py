import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T

from datasets import CargoSpaceDataset
from loss import CDB_loss
from models import Network


def validate(model, val_loader, args, device, num_classes=6):
    model.eval()
    val_total = 0
    val_correct = 0

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            out = model(val_images)
            _, val_predicted = out.max(1)

            val_total += val_labels.size(0)
            val_correct += val_predicted.eq(val_labels).sum().item()

            for i in range(len(val_predicted)):
                label = val_labels[i].item()
                pred = val_predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    class_wise_accuracy = np.divide(
        class_correct, class_total,
        out=np.zeros_like(class_correct), where=class_total != 0,
    )
    overall_accuracy = val_correct / val_total if val_total > 0 else 0.0

    return class_wise_accuracy, overall_accuracy


def train(model, train_loader, val_loader, args, device):
    optimizer = AdamW(model.parameters(), lr=3.5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160, 210], gamma=0.01)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.loss == 'cdb':
        criterion = CDB_loss(class_difficulty=np.ones(6), tau=args.tau, reduction='mean')

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_accuracy = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        total_train_samples = 0

        for train_images, train_labels in train_loader:
            train_images, train_labels = train_images.to(device), train_labels.to(device)

            optimizer.zero_grad()
            out = model(train_images)
            loss_val = criterion(out, train_labels)

            loss_val.backward()
            optimizer.step()

            epoch_loss += loss_val.item() * train_labels.size(0)
            total_train_samples += train_labels.size(0)

        scheduler.step()
        avg_train_loss = epoch_loss / total_train_samples
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.4f}")

        class_wise_accuracy, val_accuracy = validate(model, val_loader, args, device)
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        if args.loss == 'cdb':
            criterion = CDB_loss(class_difficulty=1 - class_wise_accuracy, tau=args.tau, reduction='mean')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_path = os.path.join(
                args.save_dir,
                f'best_model_data{args.dataset}_{args.attention}_{args.loss}_v2_aug.pth',
            )
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model with accuracy: {best_val_accuracy:.4f}")


def get_args():
    parser = argparse.ArgumentParser(description="Cargo Space Training Configuration")

    parser.add_argument('-d', '--dataset', type=int, choices=[0, 1], required=True,
                        help="0: 80% front | 1: 80% front+back")
    parser.add_argument('--data_dir', type=str, default='Cargo space', help="Root directory of the dataset")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('-a', '--attention', type=str, choices=['none', 'dam'], required=True, help="Attention mechanism")
    parser.add_argument('-l', '--loss', type=str, choices=['cdb', 'ce'], required=True, help="Loss function")
    parser.add_argument('--epochs', type=int, default=50, help="Number of max epochs")
    parser.add_argument('--tau', type=str, default='dynamic', help="Tau for CDB loss")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help="Model save directory")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device} with Args: Dataset={args.dataset}, Attention={args.attention}, Loss={args.loss}")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CargoSpaceDataset(root_dir=args.data_dir, dataset_type=args.dataset, mode='train', transform=transform)
    train_dataset_aug = CargoSpaceDataset(root_dir=args.data_dir, dataset_type=args.dataset, mode='train', transform=train_transform)
    train_dataset = ConcatDataset([train_dataset, train_dataset_aug])
    test_dataset = CargoSpaceDataset(root_dir=args.data_dir, dataset_type=args.dataset, mode='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Total training images: {len(train_dataset)}")
    print(f"Total testing images: {len(test_dataset)}")

    model = Network(backbone=None, attention=args.attention, class_dim=6).to(device)

    train(model, train_loader, test_loader, args, device)
