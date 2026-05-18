"""Fine-tune EfficientNet-B0 on TrashNet with two-phase training."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.utils import get_device, set_seed, MODELS_DIR
from src.preprocessing import get_dataloaders
from src.deep_model import WasteClassifierCNN
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_per_class_f1, plot_training_curves


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in tqdm(loader, leave=False, desc="  train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * len(labels)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = len(loader.dataset)
    return total_loss / n, correct / n, all_preds, all_labels


def run_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
              epochs, best_val_loss, patience, device, models_dir, phase_name):
    history = {"tr_loss": [], "vl_loss": [], "tr_acc": [], "vl_acc": []}
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["tr_loss"].append(tr_loss)
        history["vl_loss"].append(vl_loss)
        history["tr_acc"].append(tr_acc)
        history["vl_acc"].append(vl_acc)

        print(f"[{phase_name}] Ep {epoch:02d} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), f"{models_dir}/efficientnet_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    return history, best_val_loss


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 on TrashNet")
    parser.add_argument("--data_dir", default="data/raw/dataset-resized")
    parser.add_argument("--epochs_frozen", type=int, default=5)
    parser.add_argument("--epochs_unfreeze", type=int, default=25)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(args.data_dir)
    print(f"Classes: {class_names}")

    model = WasteClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    Path(MODELS_DIR).mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # Phase 1: head only
    model.freeze_backbone()
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    sched1 = CosineAnnealingLR(opt1, T_max=args.epochs_frozen)
    print("\n=== Phase 1: Training head only ===")
    h1, best_vl = run_phase(model, train_loader, val_loader, criterion,
                             opt1, sched1, args.epochs_frozen,
                             float("inf"), args.patience, device, MODELS_DIR, "P1")

    # Phase 2: fine-tune all
    model.unfreeze_backbone()
    opt2 = AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    sched2 = CosineAnnealingLR(opt2, T_max=args.epochs_unfreeze)
    print("\n=== Phase 2: Fine-tuning all layers ===")
    h2, best_vl = run_phase(model, train_loader, val_loader, criterion,
                             opt2, sched2, args.epochs_unfreeze,
                             best_vl, args.patience, device, MODELS_DIR, "P2")

    # Load best and evaluate on test set
    model.load_state_dict(torch.load(f"{MODELS_DIR}/efficientnet_best.pth", map_location=device))
    _, test_acc, preds, labels = eval_epoch(model, test_loader, criterion, device)
    m = compute_metrics(labels, preds)
    print(f"\n=== Test Metrics ===")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Macro F1 : {m['macro_f1']:.4f}")
    print(m["report"])

    # Save plots
    all_tr_loss = h1["tr_loss"] + h2["tr_loss"]
    all_vl_loss = h1["vl_loss"] + h2["vl_loss"]
    all_tr_acc  = h1["tr_acc"]  + h2["tr_acc"]
    all_vl_acc  = h1["vl_acc"]  + h2["vl_acc"]
    plot_training_curves(all_tr_loss, all_vl_loss, all_tr_acc, all_vl_acc,
                         save_path="results/efficientnet_training_curves.png")
    plot_confusion_matrix(m["confusion_matrix"],
                          save_path="results/efficientnet_confusion_matrix.png")
    plot_per_class_f1(labels, preds,
                      save_path="results/efficientnet_per_class_f1.png")

    # Save metrics to JSON
    metrics_out = {k: v for k, v in m.items() if k != "confusion_matrix" and k != "report"}
    metrics_out["confusion_matrix"] = m["confusion_matrix"].tolist()
    with open("results/efficientnet_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("Results saved to results/")


if __name__ == "__main__":
    main()
