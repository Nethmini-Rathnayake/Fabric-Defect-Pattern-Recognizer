"""Train EfficientNet-B4 for fabric defect classification."""

import argparse
import yaml
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import mlflow

from src.detection.model import FabricDefectModel
from src.detection.dataset import FabricDefectDataset, get_train_transforms, get_val_transforms


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FabricDefectDataset(cfg["data"]["train_dir"], get_train_transforms(cfg["data"]["image_size"]))
    val_ds = FabricDefectDataset(cfg["data"]["val_dir"], get_val_transforms(cfg["data"]["image_size"]))
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    model = FabricDefectModel(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = torch.cuda.amp.GradScaler() if cfg["training"]["mixed_precision"] and device.type == "cuda" else None

    mlflow.set_experiment(cfg["output"]["mlflow_experiment"])
    with mlflow.start_run():
        mlflow.log_params({k: v for section in cfg.values() if isinstance(section, dict) for k, v in section.items()})
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
            scheduler.step()

            print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc}, checkpoint_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= cfg["training"]["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)
