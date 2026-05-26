"""
Extensao inteligente de extremidades soltas (pontas) do esqueleto viario.

Filosofia:
  Uma rua NAO termina no meio de um quarteirao. Ela so pode terminar em:
    1. Outra estrada (juncao)
    2. Borda da imagem (truncamento valido)
    3. Beco sem saida / fim natural (cul-de-sac)

Se uma ponta nao satisfaz nenhuma dessas condicoes, ou:
  (a) extendemos a ponta ate encontrar uma das condicoes validas, ou
  (b) removemos a ponta (era ruido).

O raciocinio decide entre (a) e (b) baseado em quao plausivel e a
continuacao (cor, road_prob, ausencia de obstaculos).

Obstaculos para uma rua:
  - Telhado (cor saturada uniforme em poligono regular)
  - Vegetacao densa (verde dominante)
  - Pilha de terra/morro (variacao brusca de elevacao implicada pela cor)
"""

import cv2
import math
import numpy as np
from collections import deque


# -------------------------------------------------------------------------
# Helpers de baixo nivel
# -------------------------------------------------------------------------

DIRS8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def _skel_endpoints_and_junctions(skel):
    """Retorna (endpoints_yx, junctions_mask)."""
    s = (skel > 0).astype(np.uint8)
    k8 = np.ones((3, 3), np.uint8); k8[1, 1] = 0
    nb = cv2.filter2D(s, -1, k8)
    eps = np.argwhere(s & (nb == 1))
    jct = (s & (nb >= 3)).astype(np.uint8)
    return eps, jct


def _local_tangent(skel, sy, sx, look_back=12):
    """
    Retorna vetor unitario (dy,dx) apontando para FORA da via existente
    a partir do endpoint (sy,sx). Tambem retorna a curva angular media
    (rad/passo) nos ultimos passos do esqueleto, para extrapolar curvas.
    """
    h, w = skel.shape
    s = (skel > 0)
    path = [(sy, sx)]
    visited = {(sy, sx)}
    cy, cx = sy, sx
    for _ in range(look_back):
        # parar se chegamos numa juncao
        nc = 0
        for dy, dx in DIRS8:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and s[ny, nx]:
                nc += 1
        if nc >= 3 and len(path) > 1:
            break
        nxt = None
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and s[ny, nx] and (ny, nx) not in visited:
                nxt = (ny, nx)
                break
        if nxt is None:
            break
        visited.add(nxt)
        path.append(nxt)
        cy, cx = nxt

    if len(path) < 3:
        return None, 0.0

    pts = np.array(path, dtype=np.float32)  # path[0] = endpoint, path[-1] = far
    end_y, end_x = pts[0]
    far_y, far_x = pts[-1]
    vy, vx = end_y - far_y, end_x - far_x
    n = math.hypot(vy, vx)
    if n < 0.5:
        return None, 0.0
    tan = (vy / n, vx / n)

    if len(pts) >= 6:
        mid = len(pts) // 2
        v1y, v1x = pts[mid][0] - pts[-1][0], pts[mid][1] - pts[-1][1]
        v2y, v2x = pts[0][0] - pts[mid][0], pts[0][1] - pts[mid][1]
        n1, n2 = math.hypot(v1y, v1x), math.hypot(v2y, v2x)
        if n1 > 0 and n2 > 0:
            cos_a = np.clip((v1y*v2y + v1x*v2x) / (n1 * n2), -1.0, 1.0)
            ang = math.acos(cos_a)
            if v1y * v2x - v1x * v2y < 0:
                ang = -ang
            per_step = ang / max(mid, 1)
            # limitar curvatura para nao gerar caracois
            per_step = float(np.clip(per_step, -0.20, 0.20))
            return tan, per_step

    return tan, 0.0


def _rotate(vy, vx, rad):
    ca, sa = math.cos(rad), math.sin(rad)
    return (vy * ca - vx * sa, vy * sa + vx * ca)


# -------------------------------------------------------------------------
# Classificadores rapidos de "obstaculo"
# -------------------------------------------------------------------------

def _is_dense_vegetation_pixel(g, r, b):
    # Mais sensivel: pega folhagem leve tambem. Asfalto cinza tem g~r~b,
    # entao essa regra so dispara em pixels REALMENTE esverdeados.
    return (g - r > 12) and (g - b > 8)


def _is_vegetation_region(img_rgb, y, x, radius=4, min_frac=0.55):
    """Verifica se a vizinhanca (raio) tem maioria de pixels verdes."""
    h, w = img_rgb.shape[:2]
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    region = img_rgb[y0:y1, x0:x1]
    g = region[:, :, 1].astype(np.int32)
    r = region[:, :, 0].astype(np.int32)
    b = region[:, :, 2].astype(np.int32)
    green = ((g - r > 12) & (g - b > 8))
    return (green.sum() / max(green.size, 1)) >= min_frac


def _is_roof_like(img_rgb, y, x, patch=7, min_sat=80, min_uniform=0.65):
    """
    Telhado: regiao saturada (vermelho/marrom/cinza) com cor RELATIVAMENTE
    uniforme em um patch. Distingue de asfalto (sat baixa, val medio).
    """
    h, w = img_rgb.shape[:2]
    y0, y1 = max(0, y - patch), min(h, y + patch + 1)
    x0, x1 = max(0, x - patch), min(w, x + patch + 1)
    region = img_rgb[y0:y1, x0:x1]
    hsv    = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    sat    = hsv[:, :, 1]
    if float(sat.mean()) < min_sat:
        return False
    # uniformidade: porcentagem de pixels dentro de +-15 hue do hue dominante
    hue = hsv[:, :, 0]
    h_med = int(np.median(hue))
    close = (np.abs(hue.astype(np.int32) - h_med) <= 15).sum() / max(hue.size, 1)
    return close >= min_uniform


# -------------------------------------------------------------------------
# Extensao direcional adaptativa
# -------------------------------------------------------------------------

def smart_ray_march(img_rgb, road_prob_map, road_mask, ep_y, ep_x,
                    base_dy, base_dx, curve_per_step,
                    max_steps=80, search_angles_deg=(-25, -12, 0, 12, 25),
                    color_tol=72, prob_min=0.0008, max_unsupported=22):
    """
    Anda 1 passo por iteracao, ajustando direcao via:
      - curve_per_step (curvatura herdada do segmento)
      - busca local angular pelo melhor pixel (mais brilhante em road_prob,
        cor proxima a referencia, sem obstaculo)

    Retorna (path, status):
      path   : lista [(y, x), ...] dos pixels percorridos (excluindo endpoint)
      status : "road"   - encontrou outra estrada (path[-1] ja toca road_mask)
               "border" - chegou na borda
               "veg"    - bloqueado por vegetacao densa
               "roof"   - bloqueado por estrutura tipo telhado
               "fade"   - road_prob morreu (provavel dead-end)
               "out"    - usou todos os steps sem decidir
    """
    h, w = img_rgb.shape[:2]
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    img_g = img_rgb[:, :, 1].astype(np.int32)
    img_r = img_rgb[:, :, 0].astype(np.int32)
    img_b = img_rgb[:, :, 2].astype(np.int32)

    y0, y1 = max(0, ep_y - 3), min(h, ep_y + 4)
    x0, x1 = max(0, ep_x - 3), min(w, ep_x + 4)
    ref_lab = img_lab[y0:y1, x0:x1].mean(axis=(0, 1))

    cy, cx = float(ep_y), float(ep_x)
    dy, dx = base_dy, base_dx
    path = []
    unsupported = 0

    for step in range(max_steps):
        # aplicar curvatura herdada
        dy, dx = _rotate(dy, dx, curve_per_step)

        best = None
        best_score = -1.0
        for ang in search_angles_deg:
            r = math.radians(ang)
            tdy, tdx = _rotate(dy, dx, r)
            ny = int(round(cy + tdy))
            nx = int(round(cx + tdx))
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            # vegetacao (pixel-level + regional)
            if _is_dense_vegetation_pixel(int(img_g[ny, nx]),
                                          int(img_r[ny, nx]),
                                          int(img_b[ny, nx])):
                continue
            if _is_vegetation_region(img_rgb, ny, nx, radius=3, min_frac=0.5):
                continue
            # score e probabilidade
            p = float(road_prob_map[ny, nx]) if road_prob_map is not None else 0.5
            
            cdist = float(np.linalg.norm(img_lab[ny, nx] - ref_lab))
            
            # Relaxar a tolerância de cor se a IA / bordas tiverem alta confiança
            # Isso permite que a extensão continue mesmo se a pista mudar de asfalto para terra
            local_color_tol = color_tol
            if p > 0.25:
                local_color_tol += 50
                
            if cdist > local_color_tol:
                continue
                
            score = p * 2.0 + (1.0 - cdist / max(color_tol, 1.0))
            # bonus pequeno para a direcao 0 (manter inercia)
            if ang == 0:
                score += 0.05
            if score > best_score:
                best_score = score
                best = (ny, nx, tdy, tdx, p)

        if best is None:
            # nenhuma direcao serve — provavelmente bloqueado por obstaculo
            # checar qual obstaculo predominou
            ny = int(round(cy + dy))
            nx = int(round(cx + dx))
            if 0 <= ny < h and 0 <= nx < w:
                if _is_dense_vegetation_pixel(int(img_g[ny, nx]),
                                              int(img_r[ny, nx]),
                                              int(img_b[ny, nx])):
                    return path, "veg"
                if _is_roof_like(img_rgb, ny, nx):
                    return path, "roof"
            return path, "fade"

        ny, nx, ndy, ndx, p = best
        # chegou em via existente?
        if len(path) > 1 and road_mask[ny, nx] > 0:
            path.append((ny, nx))
            return path, "road"
        path.append((ny, nx))

        # contabilizar nao-suporte
        if road_prob_map is not None and p > prob_min:
            unsupported = 0
        else:
            unsupported += 1
            if unsupported > max_unsupported:
                return path, "fade"

        cy, cx = float(ny), float(nx)
        dy, dx = ndy, ndx
        # adaptar referencia gradualmente
        ref_lab = 0.80 * ref_lab + 0.20 * img_lab[ny, nx]

        # borda
        if cy <= 2 or cx <= 2 or cy >= h - 3 or cx >= w - 3:
            return path, "border"

    return path, "out"


# -------------------------------------------------------------------------
# Pipeline de limpeza com regras de termino plausivel
# -------------------------------------------------------------------------

def extend_or_prune_stubs(surface_map, width_map, img_rgb, road_prob_map=None,
                           max_extend=80, min_stub_keep=18,
                           border_margin=8, color_tol=72,
                           iterations=2):
    """
    Para cada endpoint do esqueleto:

      1. Se proximo da borda (< border_margin px), MANTEM (truncamento valido).
      2. Estima tangente + curvatura local.
      3. Ray-march com smart_ray_march.
      4. Decide pelo status:
           - 'road'   -> bridge: pinta o caminho com a cor do segmento
           - 'border' -> bridge ate borda
           - 'fade'   -> dead-end plausivel. Se ramo do stub >= min_stub_keep,
                         mantem; senao remove.
           - 'roof'   -> termino invalido. Remove o ramo inteiro do stub
                         (rua nao termina em telhado).
           - 'veg'    -> termino plausivel se for fim de quarteirao com
                         vegetacao (parque, beco). Mantem se ramo grande,
                         remove se for stub curto que so apareceu por ruido.
           - 'out'    -> indeterminado. Remove se curto.

    Repete iterations vezes (remover um stub pode expor novo stub).
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        skeletonize = None

    def _skel(smap):
        if skeletonize is not None:
            return skeletonize(smap > 0).astype(np.uint8)
        return (smap > 0).astype(np.uint8)

    h, w = surface_map.shape
    result_map = surface_map.copy()
    result_w   = width_map.copy()

    for _round in range(iterations):
        skel = _skel(result_map)
        eps, _ = _skel_endpoints_and_junctions(skel)
        road_mask = (result_map > 0).astype(np.uint8)

        any_change = False

        for ep in eps:
            sy, sx = int(ep[0]), int(ep[1])

            # Saiu do mapa pela limpeza anterior?
            if result_map[sy, sx] == 0:
                continue

            # Borda -> manter
            if sy < border_margin or sx < border_margin or \
               sy >= h - border_margin or sx >= w - border_margin:
                continue

            stype = int(result_map[sy, sx])
            ref_w = max(float(result_w[sy, sx]), 2.0)

            tan, curve = _local_tangent(skel, sy, sx, look_back=12)
            if tan is None:
                continue
            base_dy, base_dx = tan

            path, status = smart_ray_march(
                img_rgb, road_prob_map, road_mask,
                sy, sx, base_dy, base_dx, curve,
                max_steps=max_extend, color_tol=color_tol
            )

            if status in ("road", "border") and len(path) > 0:
                # Bridge
                for (py, px) in path:
                    if result_map[py, px] == 0:
                        result_map[py, px] = stype
                        result_w[py, px]   = ref_w
                road_mask[:] = (result_map > 0).astype(np.uint8)
                any_change = True
                continue

            # Caso contrario: avaliar comprimento do ramo da extremidade
            branch_len = _branch_length(skel, sy, sx)

            if status == "fade":
                if branch_len < min_stub_keep:
                    _erase_branch(result_map, result_w, skel, sy, sx)
                    any_change = True
                # senao mantem (dead-end natural)
            elif status == "roof":
                # Nunca aceita: rua nao termina em telhado.
                # Se o ramo for grande, tenta com cone mais largo antes de apagar.
                if branch_len < min_stub_keep * 2:
                    _erase_branch(result_map, result_w, skel, sy, sx)
                    any_change = True
                else:
                    # ultima tentativa: cone bem aberto
                    path2, st2 = smart_ray_march(
                        img_rgb, road_prob_map, road_mask,
                        sy, sx, base_dy, base_dx, curve,
                        max_steps=int(max_extend * 1.5),
                        search_angles_deg=(-50, -30, -15, 0, 15, 30, 50),
                        color_tol=color_tol + 12
                    )
                    if st2 in ("road", "border") and len(path2) > 0:
                        for (py, px) in path2:
                            if result_map[py, px] == 0:
                                result_map[py, px] = stype
                                result_w[py, px]   = ref_w
                        any_change = True
                    else:
                        _erase_branch(result_map, result_w, skel, sy, sx)
                        any_change = True
            elif status == "veg":
                # Termino em vegetacao: se ramo grande, e fim de quarteirao;
                # se curto, e mancha — apaga
                if branch_len < min_stub_keep:
                    _erase_branch(result_map, result_w, skel, sy, sx)
                    any_change = True
            else:  # 'out'
                if branch_len < min_stub_keep:
                    _erase_branch(result_map, result_w, skel, sy, sx)
                    any_change = True

        if not any_change:
            break

    return result_map, result_w


def _branch_length(skel, sy, sx, max_steps=200):
    """Comprimento de um ramo a partir de uma extremidade ate juncao/fim."""
    h, w = skel.shape
    s = (skel > 0)
    k8 = np.ones((3, 3), np.uint8); k8[1, 1] = 0
    nb = cv2.filter2D(skel.astype(np.uint8), -1, k8)

    visited = {(sy, sx)}
    cy, cx = sy, sx
    n = 1
    for _ in range(max_steps):
        nxt = None
        for dy, dx in DIRS8:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and s[ny, nx] and (ny, nx) not in visited:
                if int(nb[ny, nx]) >= 3:
                    return n  # juncao
                nxt = (ny, nx); break
        if nxt is None:
            return n
        visited.add(nxt)
        cy, cx = nxt
        n += 1
    return n


def _erase_branch(surface_map, width_map, skel, sy, sx, max_steps=200):
    """Apaga o ramo (no surface/width maps) a partir do endpoint ate a juncao."""
    h, w = surface_map.shape
    s = (skel > 0)
    k8 = np.ones((3, 3), np.uint8); k8[1, 1] = 0
    nb = cv2.filter2D(skel.astype(np.uint8), -1, k8)

    branch = [(sy, sx)]
    visited = {(sy, sx)}
    cy, cx = sy, sx
    for _ in range(max_steps):
        nxt = None
        for dy, dx in DIRS8:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and s[ny, nx] and (ny, nx) not in visited:
                if int(nb[ny, nx]) >= 3:
                    nxt = None; break
                nxt = (ny, nx); break
        if nxt is None:
            break
        visited.add(nxt)
        branch.append(nxt)
        cy, cx = nxt

    for (py, px) in branch:
        r = max(1, min(int(round(float(width_map[py, px]) / 2.0)), 10))
        ry0, ry1 = max(0, py - r), min(h, py + r + 1)
        rx0, rx1 = max(0, px - r), min(w, px + r + 1)
        surface_map[ry0:ry1, rx0:rx1] = 0
        width_map[ry0:ry1, rx0:rx1] = 0
