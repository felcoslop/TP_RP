"""
soft-clDice — loss topologica para estruturas tubulares (Shit et al., CVPR 2021).

Preserva a CONECTIVIDADE do esqueleto da predicao, atacando diretamente o
sintoma "estrada que para na metade". Uso no fine-tuning do SAM-Road
(ver training/finetune_local.md):

    mask_loss = mask_loss + 0.3 * soft_cldice_loss(road_pred, road_gt)

onde road_pred/road_gt sao probabilidades/labels do CANAL DE VIA em [0,1],
shape [B, H, W] ou [B, 1, H, W].
"""

import torch
import torch.nn.functional as F


def _soft_erode(img):
    if img.dim() == 3:
        img = img.unsqueeze(1)
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img):
    if img.dim() == 3:
        img = img.unsqueeze(1)
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def soft_skel(img, iters=5):
    """Esqueletizacao diferenciavel por erosoes/aberturas iterativas."""
    if img.dim() == 3:
        img = img.unsqueeze(1)
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(pred, target, iters=5, smooth=1.0):
    """
    pred, target: probabilidades em [0,1], shape [B,H,W] ou [B,1,H,W].
    Retorna 1 - clDice (escalar; menor = melhor conectividade).
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    skel_pred = soft_skel(pred, iters)
    skel_true = soft_skel(target, iters)
    tprec = ((skel_pred * target).sum() + smooth) / (skel_pred.sum() + smooth)
    tsens = ((skel_true * pred).sum() + smooth) / (skel_true.sum() + smooth)
    cl_dice = 2.0 * tprec * tsens / (tprec + tsens)
    return 1.0 - cl_dice
