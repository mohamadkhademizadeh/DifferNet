import argparse, os, yaml, torch, numpy as np
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from utils.train_utils import build_backbone, extract_feat
from utils.flow_blocks import RealNVP

def main(args):
    cfg = yaml.safe_load(open(args.config,'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess = T.Compose([T.Resize((cfg['eval']['img_size'], cfg['eval']['img_size'])),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    img = Image.open(args.image).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)

    backbone, d = build_backbone()
    flow = RealNVP(d, n_blocks=cfg['model']['flow_blocks'], hidden=cfg['model']['hidden']).to(device)
    flow.load_state_dict(torch.load(args.model, map_location=device)['flow'])
    flow.eval(); backbone=backbone.to(device).eval()

    with torch.no_grad():
        f = extract_feat(x, backbone)
        z, log_px = flow(f)

    plt.figure()
    plt.title(f"NLL: {-log_px.item():.3f}")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="models/differnet.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    main(args)
