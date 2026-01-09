import os
import io
import torch
import streamlit as st
from PIL import Image
import numpy as np
import cv2

from models.unet import TinyUNet
from utils.color import lab_to_rgb_tensor


@st.cache_resource
def load_model(checkpoint_path: str, device: str):
    model = TinyUNet(in_channels=1, out_channels=2, base_c=32)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def pil_to_L(pil_img: Image.Image, image_size: int) -> torch.Tensor:
    img = pil_img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    img_np = np.array(img)
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0:1] / 255.0
    L_t = torch.from_numpy(L.transpose(2, 0, 1)).unsqueeze(0).float()
    return L_t


def main():
    st.set_page_config(page_title="Simple Colorizer", layout="centered")
    st.title("Simple Grayscale Colorizer")
    st.write("Upload a grayscale or color image. The app converts to L and predicts AB.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"Device: {device}")

    ckpt = st.sidebar.text_input("Checkpoint path", value="checkpoints/model_epoch_20.pth")
    image_size = st.sidebar.slider("Image size", min_value=128, max_value=512, value=256, step=64)

    model = load_model(ckpt, device)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]) 
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(img, use_container_width=True)

        with st.spinner("Colorizing..."):
            L = pil_to_L(img, image_size).to(device)
            with torch.no_grad():
                pred_ab = model(L)
                rgb = lab_to_rgb_tensor(L, pred_ab).clamp(0, 1)
            rgb_np = (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            out_img = Image.fromarray(rgb_np)

        with col2:
            st.subheader("Colorized")
            st.image(out_img, use_container_width=True)

        # Diagnostics: tensor ranges
        with st.expander("Debug info", expanded=False):
            l_min = float(L.min().cpu())
            l_max = float(L.max().cpu())
            ab_min = [float(pred_ab[:, i, :, :].min().cpu()) for i in range(2)]
            ab_max = [float(pred_ab[:, i, :, :].max().cpu()) for i in range(2)]
            rgb_min = [float(rgb[:, i, :, :].min().cpu()) for i in range(3)]
            rgb_max = [float(rgb[:, i, :, :].max().cpu()) for i in range(3)]
            st.write({
                "L_min_max": [l_min, l_max],
                "ab_min_max": [ab_min, ab_max],
                "rgb_min_max": [rgb_min, rgb_max],
            })

        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        st.download_button("Download", data=buf.getvalue(), file_name="colorized.png", mime="image/png")


if __name__ == "__main__":
    main()


