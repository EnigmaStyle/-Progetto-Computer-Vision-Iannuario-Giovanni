"""Train HOG+SVM classifier on TrashNet and evaluate on test split."""
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import GridSearchCV

from src.utils import DATA_DIR, CLASS_NAMES, set_seed
from src.features import extract_hog
from src.classical_model import WasteClassifierSVM
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_per_class_f1


def load_images_and_labels(data_dir: str):
    images, labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = Path(data_dir) / cls
        if not cls_dir.exists():
            print(f"  Warning: class folder '{cls}' not found in {data_dir}")
            continue
        paths = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg"))
        for img_path in paths:
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                images.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"  Skipping {img_path}: {e}")
    return images, np.array(labels)


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Train HOG+SVM waste classifier")
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--grid_search", action="store_true",
                        help="Run GridSearchCV for hyperparameter tuning")
    args = parser.parse_args()

    print(f"Loading images from {args.data_dir} ...")
    images, labels = load_images_and_labels(args.data_dir)
    print(f"Loaded {len(images)} images across {len(set(labels))} classes")

    print("Extracting HOG features ...")
    X = np.stack([extract_hog(img) for img in tqdm(images, desc="HOG")])

    # Stratified split: 70% train, 15% val, 15% test
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, labels, test_size=0.15, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    if args.grid_search:
        print("Running GridSearchCV ...")
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(probability=True, random_state=42))])
        param_grid = {"svm__C": [1, 10, 100], "svm__gamma": ["scale", "auto"]}
        gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=2)
        gs.fit(X_train, y_train)
        print(f"Best params: {gs.best_params_}")
        clf = WasteClassifierSVM()
        clf._pipeline = gs.best_estimator_
    else:
        clf = WasteClassifierSVM()
        clf.fit(X_train, y_train)

    # Evaluate
    for split_name, X_s, y_s in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        preds = clf.predict(X_s)
        m = compute_metrics(y_s, preds)
        print(f"\n=== {split_name} Metrics ===")
        print(f"  Accuracy : {m['accuracy']:.4f}")
        print(f"  Macro F1 : {m['macro_f1']:.4f}")
        print(m["report"])

    saved = clf.save()
    print(f"\nModel saved to {saved}")

    Path("results").mkdir(exist_ok=True)
    preds_test = clf.predict(X_test)
    m_test = compute_metrics(y_test, preds_test)
    plot_confusion_matrix(m_test["confusion_matrix"], save_path="results/svm_confusion_matrix.png")
    plot_per_class_f1(y_test, preds_test, save_path="results/svm_per_class_f1.png")
    print("Plots saved to results/")


if __name__ == "__main__":
    main()
