"""Downloads pre-trained models from Google Drive if not already present."""
import os
import sys

MODELS_DIR = "models"

_MODELS = {
    "efficientnet_best.pth": "1xjJkMTWonoKTcgPENSzlxuJfOw_osq06",
    "svm_hog.joblib":        "1emxzx2Pr8sEhfAxo9o56O0DL0YspypWX",
}


def ensure_models():
    """Download any missing model files from Google Drive."""
    missing = [
        (name, fid)
        for name, fid in _MODELS.items()
        if not os.path.exists(os.path.join(MODELS_DIR, name))
    ]
    if not missing:
        return

    try:
        import gdown
    except ImportError:
        print("Errore: installa le dipendenze con  pip install -r requirements.txt")
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, fid in missing:
        dest = os.path.join(MODELS_DIR, name)
        print(f"Download {name} da Google Drive...")
        url = f"https://drive.google.com/uc?id={fid}"
        gdown.download(url, dest, quiet=False)
        print(f"{name} pronto.")
