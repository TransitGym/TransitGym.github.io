import numpy as np
import torch

def normal_cdf(value, loc=0., scale=1.):
    return 0.5 * (1 + torch.erf((value - loc) / scale / np.sqrt(2)))


def normal_icdf(value, loc=0., scale=1.):
    return loc + scale * torch.erfinv(2 * value - 1) * np.sqrt(2)


def normal_pdf(value, loc=0., scale=1.):
    return torch.exp(-(value - loc)**2 / (2 * scale**2)) / scale / np.sqrt(2 * np.pi)

# reference: https://github.com/xtma/dsac/blob/73cb82150db7788d322b421ea4709a14f6913f4b/rlkit/torch/dsac/risk.py#L36
def distortion_de(tau, mode="n", param=0., eps=1e-8):
    # Get risk weight from derivative of Risk distortion function
    tau = tau.clamp(0., 1.)
    param = 0.75
    if mode == "n": # think nothing about self contribution / foxi
        w = torch.ones_like(tau)
    elif mode == "cf": # confident on self contribution / pride manner
        w = normal_pdf(normal_icdf(1.-tau) + param) / (normal_pdf(normal_icdf(1.-tau)) + eps)
    elif mode == "ucf":
         # unconfident on self contribution/ modest manner
        w = normal_pdf(normal_icdf(tau) + param) / (normal_pdf(normal_icdf(tau)) + eps)
    # w = w/torch.sum(w,1,keepdim=True)[0]
    return w.clamp(0., 5.)
