import argparse, os, yaml, torch
from torch.utils.data import DataLoader
from utils.dataset import ImageFolderFlat
from utils.train_utils import build_backbone, extract_feat
from utils.flow_blocks import RealNVP

def main(args):
    cfg = yaml.safe_load(open(args.config,'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = ImageFolderFlat(args.train_dir, img_size=cfg['train']['img_size'])
    dl = DataLoader(ds, batch_size=cfg['train']['batch'], shuffle=True, num_workers=2)

    backbone, d = build_backbone()
    flow = RealNVP(d, n_blocks=cfg['model']['flow_blocks'], hidden=cfg['model']['hidden']).to(device)
    backbone = backbone.to(device).eval()

    opt = torch.optim.AdamW(flow.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    for epoch in range(cfg['train']['epochs']):
        flow.train()
        tot=0.0; n=0
        for x, _ in dl:
            x = x.to(device)
            with torch.no_grad():
                f = extract_feat(x, backbone)
            z, log_px = flow(f)
            loss = -log_px.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()*x.size(0); n += x.size(0)
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} - nll={tot/n:.4f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'flow': flow.state_dict()}, args.out)
    print('Saved', args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--out", default="models/differnet.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    main(args)
