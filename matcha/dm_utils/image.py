import torch


def img_grad(img):
    # img has shape (..., H, W)
    grad_x = img[..., 1:, :-1] - img[..., :-1, :-1]
    grad_y = img[..., :-1, 1:] - img[..., :-1, :-1]
    # Result has shape (..., 2, H, W)
    return torch.cat([grad_x.unsqueeze(-3), grad_y.unsqueeze(-3)], dim=-3)


def img_hessian(img):
    _grad = img_grad(img)  # Has shape (..., 2, H, W)
    _hessian = img_grad(_grad)  # Has shape (..., 2, 2, H, W)
    return _hessian
