\
import torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.cm as cm

def _normalize(x):
    x = x - x.min()
    return x / (x.max() + 1e-6)

def grad_cam(model, img: Image.Image, img_size=224, device="cpu", target_layer=None):
    model.eval()
    tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    x = tf(img).unsqueeze(0).to(device)
    x.requires_grad_(True)

    if target_layer is None:
        tl = getattr(model, "features", None)
        if tl is not None:
            target_layer = tl[-1][0] if hasattr(tl[-1], "__getitem__") else tl[-1]
        else:
            target_layer = model.layer4[-1]

    acts, grads = [], []
    def fwd_hook(_, __, out): acts.append(out)
    def bwd_hook(_, gin, gout): grads.append(gout[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(x)
    pred = int(logits.argmax(1).item())
    score = logits[0, pred]
    model.zero_grad(set_to_none=True)
    score.backward()

    A = acts[-1]
    if A.dim() == 4: A = A[0]
    G = grads[-1]
    if G.dim() == 4: G = G[0]

    if G.shape[0] == A.shape[0]:
        w = G.mean(dim=(1,2))
        cam = (w[:,None,None]*A).sum(0)
    else:
        cam = A.mean(0)

    cam = F.relu(cam)[None,None,...]
    cam = F.interpolate(cam, size=(img_size,img_size), mode='bilinear', align_corners=False)[0,0]
    cam = _normalize(cam).detach().cpu().numpy()

    inp = x[0].detach().cpu().permute(1,2,0).numpy()
    inp = (inp - inp.min())/(inp.max()-inp.min()+1e-6)
    heat = cm.jet(cam)[..., :3]
    overlay = np.clip(0.6*inp + 0.4*heat, 0, 1)
    return {"pred": pred, "overlay": overlay, "cam": cam}
