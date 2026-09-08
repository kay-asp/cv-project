"""
ResNet-50 classifier: construction, fine-tuning, and checkpointing.

Responsibilities:
  - Build ImageNet-initialised ResNet-50 with a new classification head
  - Training loop with validation
  - Save / load checkpoints
  - Build the control model (trained on all depths)
"""

import config


def build_classifier(num_classes, pretrained=config.PRETRAINED):
    """Create a ResNet-50 with a fresh classification head.

    Args:
        num_classes: Number of species classes.
        pretrained: Whether to use ImageNet weights.

    Returns:
        torch.nn.Module
    """
    raise NotImplementedError


def train(model, train_loader, val_loader, epochs=config.EPOCHS, lr=config.LEARNING_RATE):
    """Fine-tune the model on shallow-depth crops.

    Args:
        model: The classifier.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Number of training epochs.
        lr: Learning rate.

    Returns:
        Trained model, training history dict.
    """
    raise NotImplementedError


def save_checkpoint(model, path):
    """Save model weights to disk."""
    raise NotImplementedError


def load_checkpoint(model, path):
    """Load model weights from disk."""
    raise NotImplementedError


def build_control_model(num_classes):
    """Build the control classifier trained on ALL depths.

    Used to separate distribution-shift degradation from intrinsic
    difficulty of deep imagery.

    Returns:
        torch.nn.Module (untrained — call train() separately)
    """
    raise NotImplementedError
