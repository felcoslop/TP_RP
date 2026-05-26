"""
Filtros de CONTEXTO para descartar falsos positivos do modelo:

  - is_river_or_dirt_corridor: feature longa, sinuosa, flanqueada por
    vegetacao densa nos DOIS lados, sem conexao com a rede de ruas
    (degree=0 ou 1 com a maioria das estradas). Geralmente e rio,
    riacho, leito seco ou caminho de servidao.

  - has_road_like_color_along: percorre o esqueleto do componente e
    verifica se a cor mediana e compativel com asfalto/terra (cinza
    neutro ou marrom quente). Rejeita azul/verde dominantes.
"""

import cv2
import math
import numpy as np


def _component_skeleton_pixels(mask):
    try:
        from skimage.morphology import skeletonize
        return np.argwhere(skeletonize(mask > 0))
    except ImportError:
        return np.argwhere(mask > 0)


def filter_river_corridors(surface_map, img_rgb, min_length=120,
                            flank_check_radius=14,
                            min_flank_green_frac=0.55):
    """
    Remove componentes que parecem corredores naturais (rios, valas):
      - Comprimento de esqueleto >= min_length
      - Em pelo menos 55% dos pontos do esqueleto, AMBOS os lados
        (perpendiculares ao caminho local, a flank_check_radius px)
        sao vegetacao densa.
    """
    h, w = surface_map.shape
    img_g = img_rgb[:, :, 1].astype(np.int32)
    img_r = img_rgb[:, :, 0].astype(np.int32)
    img_b = img_rgb[:, :, 2].astype(np.int32)
    veg_mask = ((img_g - img_r > 18) & (img_g - img_b > 14)).astype(np.uint8)

    out = surface_map.copy()
    road_bin = (surface_map > 0).astype(np.uint8)
    n, lbl   = cv2.connectedComponents(road_bin, connectivity=8)

    removed_total = 0
    for cid in range(1, n):
        comp = (lbl == cid).astype(np.uint8)
        skel_pts = _component_skeleton_pixels(comp)
        if len(skel_pts) < min_length:
            continue

        # Amostragem de pontos do esqueleto para analise de flanco
        step = max(1, len(skel_pts) // 50)
        sampled = skel_pts[::step]

        flanked = 0
        valid   = 0
        for i, (y, x) in enumerate(sampled):
            # direcao local: vizinho anterior no sample
            if i == 0:
                ny, nx = sampled[min(i + 1, len(sampled) - 1)]
            else:
                ny, nx = sampled[i - 1]
            vy, vx = float(y - ny), float(x - nx)
            nrm = math.hypot(vy, vx)
            if nrm < 0.5:
                continue
            # perpendicular
            py, px = -vx / nrm, vy / nrm
            yL = int(round(y + py * flank_check_radius))
            xL = int(round(x + px * flank_check_radius))
            yR = int(round(y - py * flank_check_radius))
            xR = int(round(x - px * flank_check_radius))
            if not (0 <= yL < h and 0 <= xL < w and
                    0 <= yR < h and 0 <= xR < w):
                continue
            valid += 1
            if veg_mask[yL, xL] and veg_mask[yR, xR]:
                flanked += 1

        if valid == 0:
            continue
        frac = flanked / valid
        if frac >= min_flank_green_frac:
            out[lbl == cid] = 0
            removed_total += int(comp.sum())

    return out, removed_total


def filter_river_segments(surface_map, width_map, img_rgb,
                           min_segment_length=80,
                           flank_check_radius=14,
                           min_flank_green_frac=0.55):
    """
    Variante mais cirurgica: examina cada SEGMENTO entre juncoes/extremidades
    do esqueleto. Se um segmento longo for flanqueado por vegetacao em ambos
    os lados em > min_flank_green_frac dos pontos, ele e removido.

    Importante quando rios/leitos secos se conectam a estradas reais via
    pontes/atalhos — eles formariam um unico componente conectado que o filtro
    de componente inteiro nao consegue separar.
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        skeletonize = None

    h, w = surface_map.shape
    if skeletonize is None:
        return surface_map, 0

    skel = skeletonize(surface_map > 0).astype(np.uint8)
    k8 = np.ones((3, 3), np.uint8); k8[1, 1] = 0
    nb = cv2.filter2D(skel, -1, k8)
    endpoints = (skel & (nb == 1))
    junctions = (skel & (nb >= 3))

    # Marcar pixels que SAO juncao/endpoint
    vertex_mask = (endpoints | junctions).astype(np.uint8)

    img_g = img_rgb[:, :, 1].astype(np.int32)
    img_r = img_rgb[:, :, 0].astype(np.int32)
    img_b = img_rgb[:, :, 2].astype(np.int32)
    veg = ((img_g - img_r > 18) & (img_g - img_b > 14)).astype(np.uint8)

    # Visitar cada pixel do esqueleto e construir segmentos entre vertices
    DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    visited = np.zeros_like(skel, dtype=np.uint8)

    out = surface_map.copy()
    removed_total = 0

    vy, vx = np.where(vertex_mask > 0)
    seeds = list(zip(vy.tolist(), vx.tolist()))

    for sy, sx in seeds:
        for dy0, dx0 in DIRS:
            ny, nx = sy + dy0, sx + dx0
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not skel[ny, nx] or visited[ny, nx] or vertex_mask[ny, nx]:
                continue
            # Caminhar pelo esqueleto ate proximo vertice
            path = [(ny, nx)]
            visited[ny, nx] = 1
            cy, cx = ny, nx
            reached_vertex = False
            while True:
                nxt = None
                for dy, dx in DIRS:
                    yy, xx = cy + dy, cx + dx
                    if not (0 <= yy < h and 0 <= xx < w):
                        continue
                    if not skel[yy, xx] or visited[yy, xx]:
                        continue
                    if vertex_mask[yy, xx]:
                        reached_vertex = True
                        nxt = (yy, xx)
                        break
                    nxt = (yy, xx)
                    break
                if nxt is None:
                    break
                yy, xx = nxt
                if vertex_mask[yy, xx]:
                    break
                visited[yy, xx] = 1
                path.append((yy, xx))
                cy, cx = yy, xx

            if len(path) < min_segment_length:
                continue

            # Calcular flanco verde para o segmento
            flanked, valid = 0, 0
            step = max(1, len(path) // 40)
            sampled = path[::step]
            for i, (py, px) in enumerate(sampled):
                if i == 0:
                    qy, qx = sampled[min(i + 1, len(sampled) - 1)]
                else:
                    qy, qx = sampled[i - 1]
                vyf, vxf = float(py - qy), float(px - qx)
                nrm = math.hypot(vyf, vxf)
                if nrm < 0.5:
                    continue
                pdy, pdx = -vxf / nrm, vyf / nrm
                yL = int(round(py + pdy * flank_check_radius))
                xL = int(round(px + pdx * flank_check_radius))
                yR = int(round(py - pdy * flank_check_radius))
                xR = int(round(px - pdx * flank_check_radius))
                if not (0 <= yL < h and 0 <= xL < w and
                        0 <= yR < h and 0 <= xR < w):
                    continue
                valid += 1
                if veg[yL, xL] and veg[yR, xR]:
                    flanked += 1
            if valid == 0:
                continue
            if flanked / valid >= min_flank_green_frac:
                # Apagar pixels do segmento + largura local
                for (py, px) in path:
                    r = max(2, min(int(round(float(width_map[py, px]) / 2.0)), 16))
                    ry0, ry1 = max(0, py - r), min(h, py + r + 1)
                    rx0, rx1 = max(0, px - r), min(w, px + r + 1)
                    out[ry0:ry1, rx0:rx1] = 0
                removed_total += len(path)

    return out, removed_total


def filter_non_road_color(surface_map, img_rgb,
                           min_length=40,
                           max_blue_dominance=8,
                           max_green_dominance=12):
    """
    Remove componentes cujo perfil mediano de cor seja azul ou verde
    (rios, lagos, gramado). Mantem cinzas, marroms e brancos.

    Heuristica por componente:
      - mediana(B - max(R,G)) > max_blue_dominance -> agua
      - mediana(G - max(R,B)) > max_green_dominance -> vegetacao
    """
    out = surface_map.copy()
    road_bin = (surface_map > 0).astype(np.uint8)
    n, lbl   = cv2.connectedComponents(road_bin, connectivity=8)
    for cid in range(1, n):
        comp = (lbl == cid)
        if int(comp.sum()) < min_length:
            continue
        r = img_rgb[comp, 0].astype(np.int32)
        g = img_rgb[comp, 1].astype(np.int32)
        b = img_rgb[comp, 2].astype(np.int32)
        blue_dom  = int(np.median(b - np.maximum(r, g)))
        green_dom = int(np.median(g - np.maximum(r, b)))
        if blue_dom > max_blue_dominance or green_dom > max_green_dominance:
            out[comp] = 0
    return out
