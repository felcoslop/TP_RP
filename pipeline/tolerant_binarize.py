"""
Binarizacao adaptativa e tolerante do mapa de probabilidade do modelo.

Diferente do bridge_gaps.py (que usa thresholds absolutos calibrados para
imagens de alta confianca), aqui tudo e proporcional ao maximo do mapa,
o que faz funcionar tambem em imagens onde o modelo retorna probabilidades
mais baixas (zoom alto, favela, vias estreitas).

Tres modos:
  "fast"      — maximiza recall. Bom para iteracao.
  "balanced"  — recall razoavel, menos falsos positivos.
  "precise"   — pouco recall, alta precisao (similar ao bridge_gaps original).
"""

import cv2
import numpy as np


PROFILES = {
    "fast":     {"high_mult": 0.18, "low_mult": 0.03, "min_strong": 0.005,
                 "min_weak": 0.001, "close_r": 3, "min_area": 80,
                 "elong_min": 1.8},
    "balanced": {"high_mult": 0.25, "low_mult": 0.05, "min_strong": 0.008,
                 "min_weak": 0.0015, "close_r": 4, "min_area": 150,
                 "elong_min": 2.2},
    "precise":  {"high_mult": 0.32, "low_mult": 0.07, "min_strong": 0.012,
                 "min_weak": 0.003, "close_r": 4, "min_area": 220,
                 "elong_min": 2.6},
}


def _hysteresis(prob_map, low, high):
    try:
        from skimage.filters import apply_hysteresis_threshold
        return apply_hysteresis_threshold(prob_map, low=low, high=high).astype(np.uint8)
    except ImportError:
        weak   = (prob_map >= low).astype(np.uint8)
        strong = (prob_map >= high).astype(np.uint8)
        n, lbl = cv2.connectedComponents(weak, connectivity=8)
        keep   = np.zeros_like(weak)
        for lid in range(1, n):
            comp = lbl == lid
            if strong[comp].any():
                keep[comp] = 1
        return keep


def _filter_non_elongated(binary, elong_min, max_blob_factor=8.0,
                          typical_width=8.0):
    """
    Remove componentes nao alongados (telhados, manchas). Cada componente
    tem elongation calculada via PCA. Estradas tem elongation alto.
    """
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = binary.copy()
    blob_max_area = int(typical_width * typical_width * max_blob_factor)
    for lid in range(1, n):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        # vias longas mantem; blobs grandes redondos saem
        comp = (lbl == lid)
        coords = np.argwhere(comp)
        if len(coords) < 8:
            out[comp] = 0
            continue
        c = coords - coords.mean(axis=0)
        cov = np.cov(c.T)
        ev = np.linalg.eigvalsh(cov)
        ev = np.maximum(ev, 1e-6)
        elong = np.sqrt(ev[1] / ev[0])
        # se for muito grande E alongado: rede de estradas — manter
        if area > 1500:
            if elong < 1.3:
                out[comp] = 0
            continue
        if area < blob_max_area and elong < elong_min:
            out[comp] = 0
    return out


def estimate_typical_width(prob_map, low_thr=0.02):
    """Estima largura tipica de via a partir do mapa de prob."""
    bin_ = (prob_map > low_thr).astype(np.uint8) * 255
    if int((bin_ > 0).sum()) < 300:
        return 8.0
    dist = cv2.distanceTransform(bin_, cv2.DIST_L2, 5)
    return float(np.clip(np.percentile(dist[bin_ > 0], 75) * 2.0, 4.0, 60.0))


def tolerant_binarize(prob_map, profile="balanced", typical_width=None,
                      keypoint_map=None):
    """
    Pipeline:
      1. Hysteresis adaptativa (thresholds proporcionais ao max)
      2. Fechamento morfologico
      3. Filtro de elongation (remove telhados pequenos)
    Retorna mascara uint8 (0/1).
    """
    cfg = PROFILES.get(profile, PROFILES["balanced"])
    p_max = float(prob_map.max())
    if p_max < 1e-4:
        return np.zeros_like(prob_map, dtype=np.uint8)

    high = max(cfg["min_strong"], p_max * cfg["high_mult"])
    low  = max(cfg["min_weak"],   p_max * cfg["low_mult"])
    if typical_width is None:
        typical_width = estimate_typical_width(prob_map, low)

    mask = _hysteresis(prob_map, low, high)

    # Incorporar keypoints (interseccoes) como seeds adicionais
    if keypoint_map is not None:
        kp = (keypoint_map > p_max * 0.10).astype(np.uint8)
        mask = np.maximum(mask, kp)

    # Fechar gaps curtos
    r = cfg["close_r"]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Filtro de area minima
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    big = np.zeros_like(mask)
    for lid in range(1, n):
        if int(stats[lid, cv2.CC_STAT_AREA]) >= cfg["min_area"]:
            big[lbl == lid] = 1

    big = _filter_non_elongated(big, cfg["elong_min"],
                                 typical_width=typical_width)
    return big
