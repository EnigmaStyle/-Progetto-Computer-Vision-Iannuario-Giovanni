"""Gradio web app for waste classification with Grad-CAM and real-time webcam."""
import torch
import numpy as np
import cv2
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import CLASS_NAMES, get_device
from src.deep_model import WasteClassifierCNN
from src.gradcam import GradCAM
from src.preprocessing import get_transforms

DEVICE = get_device()
MODEL_PATH = "models/efficientnet_best.pth"

_ICONS = {
    "cardboard": "📦",
    "glass": "🪟",
    "metal": "🔩",
    "paper": "📄",
    "plastic": "🧴",
    "trash": "🗑️",
}


def load_model():
    model = WasteClassifierCNN()
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    except FileNotFoundError:
        print(f"Warning: {MODEL_PATH} not found — using random weights (demo mode).")
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()
grad_cam = GradCAM(model, target_layer=model.backbone[-1])
val_transform = get_transforms("val")


def _get_probs(image: np.ndarray):
    pil_img = Image.fromarray(image).convert("RGB")
    tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    return pil_img, probs


def predict(image: np.ndarray):
    if image is None:
        return None, None, "Carica un'immagine."

    pil_img, probs = _get_probs(image)
    top_idx = int(probs.argmax())

    # Grad-CAM
    tensor_grad = val_transform(pil_img).unsqueeze(0).to(DEVICE).requires_grad_(True)
    heatmap = grad_cam(tensor_grad, class_idx=top_idx)
    img_resized = np.array(pil_img.resize((224, 224)))
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.cvtColor(cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.55 * img_resized + 0.45 * heatmap_colored, 0, 255).astype(np.uint8)

    # Bar chart
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ["#e05c5c" if i == top_idx else "#4a90d9" for i in range(len(CLASS_NAMES))]
    ax.barh(CLASS_NAMES, probs * 100, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Class Probabilities")
    for i, p in enumerate(probs):
        ax.text(p * 100 + 0.5, i, f"{p:.1%}", va="center", fontsize=8)
    plt.tight_layout()

    icon = _ICONS.get(CLASS_NAMES[top_idx], "")
    label = (
        f"## {icon} {CLASS_NAMES[top_idx].upper()} — {probs[top_idx]:.1%}\n\n"
        + "\n".join(f"- **{CLASS_NAMES[i]}**: {probs[i]:.1%}" for i in probs.argsort()[::-1][:3])
    )
    return overlay, fig, label


def predict_realtime(image: np.ndarray):
    if image is None:
        return None, "In attesa della webcam..."

    _, probs = _get_probs(image)
    top_idx = int(probs.argmax())
    icon = _ICONS.get(CLASS_NAMES[top_idx], "")

    # Overlay testo sul frame
    frame = cv2.resize(image, (400, 300))
    label_text = f"{icon} {CLASS_NAMES[top_idx].upper()}  {probs[top_idx]:.1%}"
    cv2.rectangle(frame, (0, 0), (400, 40), (0, 0, 0), -1)
    cv2.putText(frame, CLASS_NAMES[top_idx].upper() + f"  {probs[top_idx]:.1%}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Barra confidenza colorata
    bar_w = int(probs[top_idx] * 380)
    color = (80, 200, 80) if probs[top_idx] > 0.7 else (200, 180, 50) if probs[top_idx] > 0.4 else (200, 80, 80)
    cv2.rectangle(frame, (10, 270), (10 + bar_w, 290), color, -1)
    cv2.rectangle(frame, (10, 270), (390, 290), (200, 200, 200), 1)

    info = (
        f"## {icon} {CLASS_NAMES[top_idx].upper()} — {probs[top_idx]:.1%}\n\n"
        + "\n".join(f"- **{CLASS_NAMES[i]}**: {probs[i]:.1%}" for i in probs.argsort()[::-1][:3])
    )
    return frame, info


with gr.Blocks(title="Waste Classifier") as demo:
    gr.Markdown("# ♻️ Waste Classifier")

    with gr.Tabs():
        # ── TAB 1: Analyze (upload + Grad-CAM) ──────────────────────────────
        with gr.Tab("Analyze"):
            gr.Markdown("Carica un'immagine o scatta una foto. Il modello mostra la predizione e la heatmap Grad-CAM.")
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(label="Input", type="numpy",
                                   sources=["upload", "webcam", "clipboard"])
                    btn = gr.Button("Classify", variant="primary")
                with gr.Column(scale=1):
                    out_overlay = gr.Image(label="Grad-CAM Overlay")
                with gr.Column(scale=1):
                    out_chart = gr.Plot(label="Confidence")
                    out_label = gr.Markdown()

            btn.click(predict, inputs=inp, outputs=[out_overlay, out_chart, out_label])
            inp.change(predict, inputs=inp, outputs=[out_overlay, out_chart, out_label])

        # ── TAB 2: Real-time webcam ──────────────────────────────────────────
        with gr.Tab("Real-time"):
            gr.Markdown("La webcam viene analizzata in tempo reale. Tieni l'oggetto davanti alla fotocamera.")
            with gr.Row():
                with gr.Column(scale=1):
                    cam = gr.Image(label="Webcam", sources=["webcam"],
                                   streaming=True, type="numpy")
                with gr.Column(scale=1):
                    rt_frame = gr.Image(label="Live prediction")
                    rt_label = gr.Markdown("In attesa della webcam...")

            cam.stream(predict_realtime, inputs=cam, outputs=[rt_frame, rt_label])

    gr.Markdown(
        "**Classi:** cardboard · glass · metal · paper · plastic · trash  \n"
        "Dataset: [TrashNet](https://github.com/garythung/trashnet) — Model: EfficientNet-B0"
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
