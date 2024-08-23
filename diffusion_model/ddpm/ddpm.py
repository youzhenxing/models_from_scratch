import torch
from torch.utils.data import DataLoader

class DDPM():
    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02
                 ):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alphas_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alphas_bars[i] = product
        self.alphas_bars = alphas_bars
        self.betas = betas
        self.alphas = alphas
        self.n_steps = n_steps
        alpha_prev = torch.empty_like(alphas_bars)
        alpha_prev[1:] = alphas_bars[0 : n_steps - 1]
        alphas_bars[0] = 1
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_prev)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alphas_bars)
    def sample_forward(self, x, t, eps = None):
        alpha_bar = self.alphas_bars[t].reshape(-1,1,1,1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

def sample_backward(self,
                    img_shape,
                    net,
                    device,
                    simple_var = True,
                    clip_x0 = True
                    ):
    x = torch.randn(img_shape).to(device)
    net = net.to(device)
    for t in range(self.n_steps -1, -1, -1):
        x = self.sample_backward_step(x, t, net, simple_var, clip_x0)
    return x

def sample_backward_step(self, x_t, t, net, simple_var = True, clip_x0 = True):
    n = x_t.shape[0]
    t_tensor = torch.tensor([t] * n, dtype= torch.long).to(x_t.device).unsqueeze(1)
    eps = net(x_t, t_tensor)

    if t == 0:
        noise = 0
    else:
      if simple_var:
         var = self.betas[t]
      else:
        var = (1 - self.alphas_bars[t - 1]) / (1 - self.alphas_bars[t]) * self.betas[t]
      noise = torch.randn_like(x_t)
      noise *= torch.sqrt(var)

    if clip_x0:
          x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) *
                 eps) / torch.sqrt(self.alpha_bars[t])
          x_0 = torch.clip(x_0, -1, 1)
          mean = self.coef1[t] * x_t + self.coef2[t] * x_0
    else:
          mean = (x_t -
                  (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                  eps) / torch.sqrt(self.alphas[t])
    x_t = mean + noise

    return x_t

21Q

def visualize_forward():
    import cv2
    import einops
    import numpy as np

    from mnist_dataset import get_dataloader

    n_stesp = 100
    device = 'mps'
    dataloader = get_dataloader(5)
    x,_ = next(iter(dataloader))

    x = x.to(device)
    ddpm = DDPM(device=device, n_steps= n_stesp)
    xts = []
    percents = torch.linspace(0, 0.9999, 30)

    for percent in percents:
        t = torch.tensor([int(n_stesp * percent)])
        print(t)
        t = t.unsqueeze(1)
        print(t)
        x_t = ddpm.sample_forward(x, t)
        xts.append(x_t)
    print(xts)
    res = torch.stack(xts,0)
    print(xts)
    res = einops.rearrange(res, 'n1 n2 c h w -> (n2 h) (n1 w) c')

    # for visualization, convert to uint8
    res = (res.clip(-1, 1) + 1) / 2 * 255

    res = res.cpu().numpy().astype(np.uint8)

    cv2.imshow('img', res)
    cv2.waitKey(0)



if __name__ == '__main__':
   visualize_forward()
