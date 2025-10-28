import numpy as np, cv2, torch

def grad_heatmap(model, backbone, x):
    # simple gradient magnitude wrt input as saliency
    x = x.clone().detach().requires_grad_(True)
    z, log_px = model(extract_flat(backbone, x))
    loss = -log_px.mean()
    loss.backward()
    g = x.grad.detach().abs().max(dim=1)[0]  # N,H,W
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    g = (g[0].cpu().numpy()*255).astype('uint8')
    return g

def extract_flat(backbone, x):
    f = backbone(x)
    return f.flatten(start_dim=1)
