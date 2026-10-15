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
    """Create a ResNet-50 with a fresh classification head.

    Args:
        num_classes: Number of species classes.
        pretrained: Whether to use ImageNet weights.

    Returns:
        torch.nn.Module
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.resnet50(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def train(model, train_loader, val_loader, epochs=config.EPOCHS, lr=config.LEARNING_RATE):
    """Fine-tune the model on shallow-depth crops.

    Args:
        model: The classifier.
        train_loader: Training DataLoader yielding (images, labels) batches,
            where labels are integer indices into the species list used to
            build the model's classification head.
        val_loader: Validation DataLoader (same contract as train_loader).
        epochs: Number of training epochs.
        lr: Learning rate.

    Returns:
        Trained model (best validation weights loaded), training history dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
                images, labels = images.to(device), labels.to(device)
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


def save_checkpoint(model, path):
    """Save model weights to disk, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    """Load model weights from disk onto the current device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


def build_control_model(num_classes):
    """Build the control classifier trained on ALL depths.

    Same architecture as the shallow-only classifier — the point of the
    control is to hold architecture and training procedure fixed and vary
    only the training distribution, so degradation differences can be
    attributed to distribution shift rather than model capacity.

    Returns:
        torch.nn.Module (untrained — call train() separately)
    """
    return build_classifier(num_classes, pretrained=config.PRETRAINED)
