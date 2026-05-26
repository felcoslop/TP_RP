"""
Deteccao de evidencias VISUAIS que confirmam asfalto: faixas brancas/amarelas,
faixas de pedestre, marcacoes elongadas em geral.

Asfalto pode existir sem marcacao, mas a presenca de marcacao e evidencia
forte de asfalto. Esta logica fortalece o sinal de asfalto onde houver
marcacao detectavel, sem penalizar asfalto sem marcacao.
"""

import cv2
import numpy as np


def detect_lane_markings(img_rgb, road_mask, min_area=8, max_area=1200,
                         min_elongation=2.4, max_thickness_px=4):
    """
    Identifica pixels que sao faixas/marcacoes viarias dentro da regiao de via.

    Heuristicas reforcadas (para evitar pegar bordas de telhado, carros brancos
    e reflexos):
      - Branco PURO (val>200, sat<35) ou amarelo (hue 20-32, sat>90, val>160)
      - Componente alongado (eixo maior / menor >= min_elongation)
      - Espessura maxima (eixo menor) <= max_thickness_px — faixas sao finas
      - Localizado ESTRITAMENTE dentro da road_mask (sem dilatacao)

    Retorna mascara binaria uint8.
    """
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_chan = hsv[:, :, 0].astype(np.int32)
    sat    = hsv[:, :, 1].astype(np.int32)
    val    = hsv[:, :, 2].astype(np.int32)

    white  = (val > 200) & (sat < 35)
    yellow = (h_chan >= 20) & (h_chan <= 32) & (sat > 90) & (val > 160)
    cand   = (white | yellow).astype(np.uint8)

    # So aceitar marcacoes ESTRITAMENTE dentro de uma estrada detectada
    rmask  = (road_mask > 0).astype(np.uint8)
    cand   = cand & rmask

    if cand.sum() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    result = np.zeros((h, w), dtype=np.uint8)
    for lid in range(1, n):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        coords = np.argwhere(lbl == lid)
        if len(coords) < 5:
            continue

        # PCA: razao entre eixos
        c   = coords - coords.mean(axis=0)
        cov = np.cov(c.T)
        ev  = np.linalg.eigvalsh(cov)
        ev  = np.maximum(ev, 1e-6)
        major, minor = float(np.sqrt(ev[1])), float(np.sqrt(ev[0]))
        elong = major / minor
        if elong < min_elongation:
            continue
        # Faixa nao pode ser grossa — limitar pelo eixo menor
        if minor > max_thickness_px:
            continue
        result[lbl == lid] = 1

    return result


def lane_marking_lock(surface_map, marking_mask, min_marks_per_segment=2,
                      target_label=1):
    """
    Para cada componente conectado de surface_map, se contiver
    min_marks_per_segment ou mais pixels de marcacao, forca a classificacao
    para asfalto (label 1).

    Marcacoes em estradas de terra sao raras — este lock e seguro.
    """
    if marking_mask.sum() == 0:
        return surface_map

    result = surface_map.copy()
    road_bin = (surface_map > 0).astype(np.uint8)
    n, lbl   = cv2.connectedComponents(road_bin, connectivity=8)

    for cid in range(1, n):
        comp = lbl == cid
        n_marks = int(marking_mask[comp].sum())
        if n_marks >= min_marks_per_segment:
            result[comp] = target_label

    return result


def detect_pedestrian_crossings(marking_mask, parallel_tol_px=4,
                                 min_stripes=3):
    """
    Faixas de pedestre = >=3 marcacoes brancas paralelas proximas e do
    mesmo tamanho. Esta funcao retorna uma mascara dilatada sobre essas
    aglomeracoes (util para forcar asfalto em interseccoes).

    Mantida intencionalmente simples: agrupa componentes brancos cuja
    bounding box maior eixo seja parecido (+-30%) e o centro esteja a
    menos de parallel_tol_px * max_stripe_length de outro.
    """
    if marking_mask.sum() == 0:
        return np.zeros_like(marking_mask)

    n, lbl, stats, cent = cv2.connectedComponentsWithStats(marking_mask, connectivity=8)
    centers = []
    sizes   = []
    for lid in range(1, n):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < 6 or area > 600:
            continue
        cy, cx = cent[lid][1], cent[lid][0]
        max_dim = max(stats[lid, cv2.CC_STAT_WIDTH], stats[lid, cv2.CC_STAT_HEIGHT])
        centers.append((cy, cx, lid))
        sizes.append(max_dim)

    if len(centers) < min_stripes:
        return np.zeros_like(marking_mask)

    h, w = marking_mask.shape
    result = np.zeros_like(marking_mask)
    used = set()
    for i, (cy, cx, lid) in enumerate(centers):
        if i in used:
            continue
        group = [i]
        for j, (cy2, cx2, lid2) in enumerate(centers):
            if j == i or j in used:
                continue
            if abs(sizes[i] - sizes[j]) > 0.4 * max(sizes[i], sizes[j]):
                continue
            d = np.hypot(cy - cy2, cx - cx2)
            if d < (sizes[i] + sizes[j]):
                group.append(j)
        if len(group) >= min_stripes:
            for k in group:
                used.add(k)
                k_lid = centers[k][2]
                result[lbl == k_lid] = 1

    # Dilatar para abranger todo o cruzamento
    result = cv2.dilate(result, np.ones((9, 9), np.uint8), iterations=2)
    return result
