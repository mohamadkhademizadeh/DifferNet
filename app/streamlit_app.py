import streamlit as st
import yaml, torch
from torchvision import transforms as T
from PIL import Image
from utils.train_utils import build_backbone, extract_feat
from utils.flow_blocks import RealNVP

st.set_page_config(page_title="DifferNet Demo", layout="wide")
st.title("🧠 DifferNet — Normalizing Flow Anomaly Detection")

with open('configs/default.yaml','r') as f:
    CFG = yaml.safe_load(f)

model_path = st.sidebar.text_input("Model path", "models/differnet.pt")
uploaded = st.file_uploader("Upload an image", type=['png','jpg','jpeg'])

if uploaded and model_path and model_path.strip():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess = T.Compose([T.Resize((CFG['eval']['img_size'], CFG['eval']['img_size'])),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    img = Image.open(uploaded).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)

    backbone, d = build_backbone()
    flow = RealNVP(d, n_blocks=CFG['model']['flow_blocks'], hidden=CFG['model']['hidden']).to(device)
    flow.load_state_dict(torch.load(model_path, map_location=device)['flow'])
    flow.eval(); backbone=backbone.to(device).eval()

    with torch.no_grad():
        f = extract_feat(x, backbone)
        z, log_px = flow(f)

    st.image(img, caption=f"NLL (anomaly score): {-log_px.item():.4f}", use_column_width=True)
else:
    st.info("Train a model first, then upload an image to score.")
