import argparse, os, yaml, glob, cv2, numpy as np, torch
from torchvision import transforms as T
from utils.train_utils import build_backbone, extract_feat
from utils.flow_blocks import RealNVP

def main(args):
    cfg = yaml.safe_load(open(args.config,'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preprocess = T.Compose([
        T.Resize((cfg['eval']['img_size'], cfg['eval']['img_size'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    backbone, d = build_backbone()
    flow = RealNVP(d, n_blocks=cfg['model']['flow_blocks'], hidden=cfg['model']['hidden']).to(device)
    flow.load_state_dict(torch.load(args.model, map_location=device)['flow'])
    flow.eval(); backbone=backbone.to(device).eval()

    os.makedirs(args.out, exist_ok=True)
    for p in glob.glob(os.path.join(args.data_dir, "*.*")):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        x = preprocess(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)[:,:,::-1])  # ensure PIL-like path
        x = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            f = extract_feat(x, backbone)
            z, log_px = flow(f)
        score = (-log_px).item()
        name = os.path.splitext(os.path.basename(p))[0]
        with open(os.path.join(args.out, f"{name}.txt"), "w") as ftxt:
            ftxt.write(str(score))
        print(name, score)

if __name__ == "__main__":
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", default="models/differnet.pt")
    ap.add_argument("--out", default="out")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    main(args)
