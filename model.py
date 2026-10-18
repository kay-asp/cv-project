"""
ResNet-50 classifier: construction, fine-tuning, and checkpointing.

Responsibilities:
  - Build ImageNet-initialised ResNet-50 with a new classification head
  - Training loop with validation
  - Save / load checkpoints
  - Build the control model (trained on all depths)
"""

import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import config


def build_classifier(num_classes, pretrained=config.PRETRAINED):
    # create ResNet-50 with fresh classification head

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.resnet50(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def train(model, train_loader, val_loader, epochs, lr, resume_path):
    # fine-tune the model on shallow-depth crops

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device, memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True
    use_amp = config.USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if config.USE_COSINE else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    start_epoch, stale_epochs = 0, 0

    if resume_path and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        history, best_val_acc, best_state = state["history"], state["best_val_acc"], state["best_state"]
        start_epoch, stale_epochs = state["epoch"] + 1, state["stale_epochs"]
        print(f"Resuming from epoch {start_epoch + 1} ({resume_path})")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += batch_size

        train_loss = running_loss / seen
        train_acc = running_correct / seen

        model.eval()
        val_loss, val_correct, val_seen = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_seen += batch_size

        val_loss /= val_seen
        val_acc = val_correct / val_seen

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if scheduler:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if resume_path:
            os.makedirs(os.path.dirname(resume_path) or ".", exist_ok=True)
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "history": history, "best_val_acc": best_val_acc, "best_state": best_state,
                "epoch": epoch, "stale_epochs": stale_epochs,
            }, resume_path)

        if stale_epochs >= config.PATIENCE:
            print(f"Early stopping: no val improvement for {config.PATIENCE} epochs")
            break

    model.load_state_dict(best_state)
    return model, history


def save_checkpoint(model, path):
    # save model weights to disk, creating parent directories if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    # load model weights from disk onto the current device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


