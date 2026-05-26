import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import deque, defaultdict


def enhance_road_mask(road_prob, fused_keypoint=None, typical_width=8.0,
                      thr_strong=None, thr_weak=None):
    """
    Preenche secoes fracas do mapa de probabilidade que conectam dois segmentos
    fortes (pontes reais) e remove secoes fracas sem conexao bilateral.

    Limiares adaptativos: derivados do maximo do road_prob para funcionar
    tanto em imagens com alta confianca quanto em imagens de alto zoom onde
    o modelo retorna confiancas menores.

    Logica:
      1. strong_bin = road_prob > thr_strong  (nucleo confiante da via)
      2. full_bin   = road_prob > thr_weak    (todo candidato a via)
      3. Para cada componente de full_bin:
           - Toca >= 2 strong_labels distintos → e uma ponte → manter tudo
           - Toca <= 1 strong_label            → nao conecta nada → manter
             apenas os pixels fortes (descarta a area fraca)
    """
    road_max = float(road_prob.max())
    if thr_strong is None:
        # 20% do maximo: adapta a confianca do modelo no zoom atual
        thr_strong = float(np.clip(road_max * 0.20, 0.006, 0.030))
    if thr_weak is None:
        thr_weak = float(np.clip(road_max * 0.03, 0.001, 0.006))

    strong_bin = (road_prob > thr_strong).astype(np.uint8)
    if fused_keypoint is not None:
        strong_bin = np.maximum(strong_bin,
                                (fused_keypoint > 0.05).astype(np.uint8))
    full_bin = (road_prob > thr_weak).astype(np.uint8)

    _, strong_labels = cv2.connectedComponents(strong_bin, connectivity=8)
    n_full, full_labels, full_stats, _ = cv2.connectedComponentsWithStats(
        full_bin, connectivity=8)

    result = np.zeros_like(strong_bin, dtype=np.uint8)
    min_area     = max(5, int(typical_width * 0.5))
    max_blob_area = int(typical_width * typical_width * 25)

    for fid in range(1, n_full):
        comp_area = int(full_stats[fid, cv2.CC_STAT_AREA])
        if comp_area < min_area:
            continue

        comp = full_labels == fid

        # Strong labels que estao dentro deste componente
        sl_vals   = strong_labels[comp & (strong_bin > 0)]
        unique_sl = set(sl_vals.tolist())
        unique_sl.discard(0)

        if len(unique_sl) >= 2:
            # Ponte bilateral: verificar se nao e um blob grande nao-elongado
            bw = int(full_stats[fid, cv2.CC_STAT_WIDTH])
            bh = int(full_stats[fid, cv2.CC_STAT_HEIGHT])
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if comp_area > max_blob_area and aspect < 1.5:
                # Blob grande → manter so nucleo forte
                result[comp & (strong_bin > 0)] = 255
            else:
                result[comp] = 255
        else:
            # Sem ponte → manter so nucleo forte
            result[comp & (strong_bin > 0)] = 255

    # Remove salt-and-pepper residual
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, k3)
    return result


def measure_road_widths(road_mask_binary):
    """
    Usa Distance Transform para medir a largura real de cada pixel de estrada.
    O valor em cada pixel = distancia ate a borda mais proxima = metade da largura.
    """
    dist = cv2.distanceTransform(road_mask_binary, cv2.DIST_L2, 5)
    return dist * 2


def classify_road_surface(img_rgb, road_mask_binary, width_map,
                          marking_mask=None):
    """
    Classifica pixels de estrada em asfalto (1) ou terra (2) com tres evidencias:

      P_cor     : HSV — hue quente (4-28) + sat > 25 → terra.
                  Pixels muito brancos/cinza-claros (val>180, sat<40) recebem
                  penalizacao forte (asfalto desgastado ou concreto).
      P_textura : desvio padrao local 7x7 em escala de cinza.
                  Terra e granular (std alto), asfalto e liso (std baixo).
      P_vizinho : Gaussian blur mascarado pela estrada propaga o tipo de
                  superficie ao longo da via (vias conectadas = mesmo tipo).

    Formula:
        p_terra_local = 0.55 * P_cor + 0.45 * P_textura
        p_final       = 0.55 * p_terra_local + 0.45 * P_vizinho
        tipo = terra se p_final >= 0.40, asfalto caso contrario

    Se marking_mask for fornecido, faixas detectadas reduzem a probabilidade
    de terra nos arredores (asfalto confirmado por marcacao).
    """
    h, w = road_mask_binary.shape
    surface_map = np.zeros((h, w), dtype=np.uint8)

    road_pixels = road_mask_binary > 0
    if not road_pixels.any():
        return surface_map

    # --- Evidencia 1: COR (HSV) ---
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue = img_hsv[:, :, 0].astype(np.float32)  # 0-179
    sat = img_hsv[:, :, 1].astype(np.float32)  # 0-255
    val = img_hsv[:, :, 2].astype(np.float32)  # 0-255

    warm_hue  = np.clip(1.0 - np.abs(hue - 14.0) / 12.0, 0.0, 1.0)
    sat_prob  = np.clip((sat - 25.0) / 100.0, 0.0, 1.0)
    p_terra_color = (warm_hue * sat_prob).astype(np.float32)

    # Asfalto claro/cinza: threshold baixado (130) para capturar asfalto medio.
    # Penaliza tanto cor quanto textura — faixas de demarcacao brancas em asfalto
    # geram alta textura mas nao sao terra.
    white_penalty = (np.clip((val - 130.0) / 80.0, 0.0, 1.0) *
                     np.clip(1.0 - sat / 60.0, 0.0, 1.0)).astype(np.float32)
    p_terra_color = p_terra_color * (1.0 - 0.90 * white_penalty)

    # Cinza neutro (sat baixo, val moderado a alto) = asfalto quase certamente
    gray_road = (np.clip(1.0 - sat / 40.0, 0.0, 1.0) *
                 np.clip((val - 100.0) / 100.0, 0.0, 1.0)).astype(np.float32)
    p_terra_color = p_terra_color * (1.0 - 0.85 * gray_road)

    # --- Evidencia 2: TEXTURA (std local 7x7) ---
    # Terra e granular (std alto), asfalto e liso (std baixo).
    # Faixas de demarcacao em asfalto causam std alto — aplicar white_penalty
    # para nao confundir textura de faixa com terra granular.
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean_l   = cv2.boxFilter(img_gray,            cv2.CV_32F, (7, 7))
    mean_sq  = cv2.boxFilter(img_gray * img_gray, cv2.CV_32F, (7, 7))
    std_map  = np.sqrt(np.maximum(mean_sq - mean_l * mean_l, 0.0))
    p_terra_texture = np.clip((std_map - 8.0) / 22.0, 0.0, 1.0).astype(np.float32)
    p_terra_texture = p_terra_texture * (1.0 - 0.70 * white_penalty)

    # --- Combinacao local ---
    p_terra_local = 0.55 * p_terra_color + 0.45 * p_terra_texture
    p_terra_local[~road_pixels] = 0.0

    # --- Evidencia 3: VIZINHANCA (Gaussian blur mascarado por largura) ---
    road_f   = road_pixels.astype(np.float32)
    w_weight = np.sqrt(np.clip(width_map, 0.5, 30.0)) * road_f
    p_num    = cv2.GaussianBlur(p_terra_local * w_weight, (81, 81), 20)
    w_den    = cv2.GaussianBlur(w_weight,                 (81, 81), 20)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_neighbor = np.where(w_den > 0.01, p_num / w_den, p_terra_local)

    # --- Probabilidade final ---
    p_final = 0.55 * p_terra_local + 0.45 * p_neighbor

    # --- Evidencia 4 (opcional): faixas/marcacoes -> reduzir p_terra ---
    # Marcacoes brancas/amarelas elongadas dentro da via SAO praticamente
    # exclusivas de asfalto. Reduzimos p_final na vizinhanca de marcacoes.
    if marking_mask is not None and marking_mask.sum() > 0:
        mk = (marking_mask > 0).astype(np.float32)
        # influencia ate ~30px ao redor da marcacao
        mk_field = cv2.GaussianBlur(mk, (61, 61), 14)
        mk_field = mk_field / max(float(mk_field.max()), 1e-6)
        p_final = p_final * (1.0 - 0.65 * mk_field)

    surface_map[road_pixels & (p_final >= 0.40)] = 2  # terra
    surface_map[road_pixels & (p_final <  0.40)] = 1  # asfalto

    return surface_map


def _skeleton_direction(skel, sy, sx, look_back=10):
    """
    Rastreia o esqueleto para tras a partir de (sy,sx) e retorna o vetor
    direcional normalizado de avanco (apontando para fora da via existente).
    Retorna None se nao conseguir estimar direcao.
    """
    h, w = skel.shape
    path = [(sy, sx)]
    visited = {(sy, sx)}
    cy, cx = sy, sx

    for _ in range(look_back):
        found = False
        # Contar vizinhos de (cy, cx) no esqueleto original (para ver se chegamos numa juncao)
        neighbors_count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] > 0:
                    neighbors_count += 1
        
        # Se for uma juncao real (>= 3 vizinhos e não é o endpoint original), PARAR
        # Caso contrario, o rastreamento invade a rua principal e a direcao fica enviesada
        if neighbors_count >= 3 and len(path) > 1:
            break

        # Priorizar direcoes cardinais depois diagonais (mais estaveis)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] > 0 and (ny, nx) not in visited:
                visited.add((ny, nx))
                path.append((ny, nx))
                cy, cx = ny, nx
                found = True
                break
        if not found:
            break

    if len(path) < 2:
        return None

    # Vetor do ponto mais distante para o endpoint = direcao de avanco
    ey, ex = path[0]
    by, bx = path[-1]
    vy, vx = float(ey - by), float(ex - bx)
    norm = np.sqrt(vy**2 + vx**2)
    if norm < 0.5:
        return None
    return (vy / norm, vx / norm)


def _measure_perp_width(img_lab, cy, cx, perp_dy, perp_dx, ref_lab, color_tol, max_half):
    """
    Mede a largura da via no ponto (cy,cx) percorrendo na direcao perpendicular
    (perp_dy, perp_dx) ate a cor divergir de ref_lab.
    Retorna a largura total em pixels (1 = so o pixel central).
    """
    h, w = img_lab.shape[:2]
    count = 1
    for sign in (1, -1):
        for d in range(1, max_half + 1):
            ny = int(round(cy + sign * d * perp_dy))
            nx = int(round(cx + sign * d * perp_dx))
            if not (0 <= ny < h and 0 <= nx < w):
                break
            if np.linalg.norm(img_lab[ny, nx] - ref_lab) > color_tol:
                break
            count += 1
    return float(count)


def trace_roads_directional(road_binary, img_rgb, surface_map, width_map,
                             road_prob_map=None,
                             max_steps=250, color_jump_max=55):
    """
    Rastreia continuacao de vias com probabilidade combinada.

    A cada passo calcula p_combined = 0.55*p_cor + 0.45*p_largura (0-1):
      p_cor    = 1 - cdist/color_jump_max  (quao similar e a cor ao perfil atual)
      p_largura = 1 - |w_local - w_ref| / max(w_local, w_ref)  (consistencia de largura)

    Momentum: contador de passos consecutivos com p_combined > 0.55.
      - Traco estabelecido (momentum >= 3) pode atravessar secoes de menor
        confianca (p >= 0.25) sem parar — representa a mudanca gradual de cor
        ao longo de uma mesma via.
      - Traco novo (momentum < 3) para em p < 0.35.

    Para em condicoes absolutas:
      - Vegetacao densa (g-r > 22 E g-b > 18)
      - Salto brusco de cor (cdist > color_jump_max)
    """
    from skimage.morphology import skeletonize

    h, w = road_binary.shape
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    img_g   = img_rgb[:, :, 1].astype(np.int32)
    img_r   = img_rgb[:, :, 0].astype(np.int32)
    img_b   = img_rgb[:, :, 2].astype(np.int32)

    skel = skeletonize(road_binary > 0).astype(np.uint8)
    k8   = np.ones((3, 3), np.uint8); k8[1, 1] = 0
    nbrs = cv2.filter2D(skel, -1, k8)
    endpoints = np.argwhere(skel & (nbrs == 1))

    result_map   = surface_map.copy()
    result_width = width_map.copy()

    ANGLES_DEG = [0, -15, 15, -30, 30, -45, 45, -60, 60]

    for ep in endpoints:
        sy, sx = int(ep[0]), int(ep[1])

        dir_vec = _skeleton_direction(skel, sy, sx, look_back=10)
        if dir_vec is None:
            continue

        cur_dy, cur_dx = dir_vec

        r0, r1 = max(0, sy - 5), min(h, sy + 6)
        c0, c1 = max(0, sx - 5), min(w, sx + 6)
        ref_lab  = img_lab[r0:r1, c0:c1].mean(axis=(0, 1))
        ref_w    = max(float(width_map[sy, sx]), 2.0)
        ref_surf = int(surface_map[sy, sx]) if surface_map[sy, sx] > 0 else 1

        cy, cx   = float(sy), float(sx)
        momentum = 0  # passos consecutivos de alta confianca

        # Limite de passos consecutivos fora do road_binary.
        # Gaps tipicos de deteccao (texto de mapa, sombra, inicio de via):
        #   vias grossas (ref_w>5): ate 35px de gap
        #   vias finas:             ate 55px de gap
        max_unsupported = 35 if ref_w > 5.0 else 60
        unsupported = 0

        for _step in range(max_steps):
            best_ny, best_nx = None, None
            best_dy, best_dx = cur_dy, cur_dx
            best_cdist       = float('inf')

            for angle_deg in ANGLES_DEG:
                if angle_deg == 0:
                    tdy, tdx = cur_dy, cur_dx
                else:
                    rad = np.radians(float(angle_deg))
                    ca, sa = np.cos(rad), np.sin(rad)
                    tdy = cur_dy * ca - cur_dx * sa
                    tdx = cur_dy * sa + cur_dx * ca

                tny = int(round(cy + tdy))
                tnx = int(round(cx + tdx))

                if not (0 <= tny < h and 0 <= tnx < w):
                    continue

                # Parada absoluta: probabilidade de estrada quase nula
                # Casas tem road_prob ~0; gaps de deteccao (sombra, texto) tem > 0.001
                if road_prob_map is not None and float(road_prob_map[tny, tnx]) < 0.001:
                    continue

                # Parada absoluta: vegetacao densa
                if img_g[tny, tnx] - img_r[tny, tnx] > 22 and \
                   img_g[tny, tnx] - img_b[tny, tnx] > 18:
                    continue

                # Parada absoluta: salto brusco de cor
                cdist = float(np.linalg.norm(img_lab[tny, tnx] - ref_lab))
                
                local_color_jump_max = color_jump_max
                if road_prob_map is not None:
                    p = float(road_prob_map[tny, tnx])
                    if p > 0.25:
                        local_color_jump_max += 60  # Tolera transições asfalto/terra
                        
                if cdist > local_color_jump_max:
                    continue

                if cdist < best_cdist:
                    best_cdist = cdist
                    best_ny, best_nx = tny, tnx
                    best_dy, best_dx = tdy, tdx

            if best_ny is None:
                break

            ny, nx = best_ny, best_nx

            if surface_map[ny, nx] > 0:
                break  # Conectou a via existente

            # --- Probabilidade combinada do passo ---
            p_cor = max(0.0, 1.0 - best_cdist / color_jump_max)

            perp_dy, perp_dx = -best_dx, best_dy
            local_w = _measure_perp_width(
                img_lab, ny, nx, perp_dy, perp_dx,
                ref_lab, color_jump_max * 0.8, max_half=int(ref_w * 2.0)
            )
            denom = max(ref_w, local_w, 1.0)
            p_largura = max(0.0, 1.0 - abs(local_w - ref_w) / denom)

            p_combined = 0.55 * p_cor + 0.45 * p_largura

            # Atualizar momentum
            if p_combined >= 0.55:
                momentum = min(momentum + 1, 10)
            else:
                momentum = max(0, momentum - 1)

            # Limiar de continuacao adaptado ao momentum:
            # Via estabelecida (momentum>=3) tolera secoes de menor confianca
            min_p = 0.18 if momentum >= 3 else 0.28
            if p_combined < min_p:
                break

            result_map[ny, nx]   = ref_surf
            result_width[ny, nx] = local_w if local_w > 1 else ref_w

            # Contar passos consecutivos sem suporte no road_binary detectado.
            # Estradas reais tem pixels detectados; espaco vazio nao tem.
            if road_binary[ny, nx] > 0:
                unsupported = 0
            else:
                unsupported += 1
                if unsupported > max_unsupported:
                    break

            cy, cx         = float(ny), float(nx)
            cur_dy, cur_dx = best_dy, best_dx

            # Referencia adaptativa: 75% historico + 25% pixel atual
            ref_lab = 0.75 * ref_lab + 0.25 * img_lab[ny, nx]

    return result_map, result_width




def unify_segment_surface(surface_map):
    """
    Uma estrada continua nao pode ter dois pavimentos.
    Para cada componente conectado do surface_map, aplica votacao por maioria:
      - >= 65% asfalto  → tudo asfalto
      - >= 65% terra    → tudo terra
      - caso contrario  → mantém (junção real entre dois tipos)
    """
    result = surface_map.copy()
    road_mask = (surface_map > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(road_mask, connectivity=8)

    for lid in range(1, num_labels):
        comp = labels == lid
        n_asf = int((surface_map[comp] == 1).sum())
        n_ter = int((surface_map[comp] == 2).sum())
        total = n_asf + n_ter
        if total == 0:
            continue
        if n_asf >= total * 0.65:
            result[comp] = 1
        elif n_ter >= total * 0.65:
            result[comp] = 2

    return result


def filter_non_road_shapes(surface_map, min_aspect=2.5, min_area=30):
    """
    Remove componentes conectados que nao tem formato de estrada.
    Estradas sao formas alongadas (retas ou com curvas suaves).
    Circulos, blobs e splashes tem aspect ratio proximo de 1 — sao descartados.

    Regras por componente:
      1. Area < min_area px  → ruido, remove.
      2. Toca outro componente de estrada (dentro de 7px) em mais de 15px
         → provavelmente juncao/extremidade, manter.
      3. max_lado / min_lado do rect. minimo envolvente < min_aspect → remove.
    """
    h, w = surface_map.shape
    result = surface_map.copy()

    road_mask = (surface_map > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(road_mask, connectivity=8)

    dilate_k = np.ones((7, 7), np.uint8)

    for lid in range(1, num_labels):
        comp = (labels == lid).astype(np.uint8)
        area = int(comp.sum())

        # Ruido minimo
        if area < min_area:
            result[comp > 0] = 0
            continue

        # Componentes grandes sao quase certamente estradas reais
        # (redes com juncoes tem forma complexa nao-elongada — nao aplicar aspect ratio)
        if area >= 600:
            continue

        # Verificar conexao com outros componentes de estrada
        other_roads = road_mask.copy()
        other_roads[comp > 0] = 0
        expanded = cv2.dilate(comp, dilate_k)
        touching = int(np.logical_and(expanded > 0, other_roads > 0).sum())
        if touching > 15:
            continue

        # Analise de formato: rect. minimo envolvente (so para componentes pequenos isolados)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            result[comp > 0] = 0
            continue

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]

        if rh < 0.5 or rw < 0.5:
            result[comp > 0] = 0
            continue

        aspect = max(rw, rh) / min(rw, rh)
        if aspect < min_aspect:
            result[comp > 0] = 0

    return result


def close_road_gaps(surface_map, width_map, skeleton, img_rgb,
                    road_prob_map=None, max_gap=55, color_tol=65):
    """
    Para cada extremidade do esqueleto varre multiplas direcoes ate max_gap passos.
    Se encontrar um pixel de estrada existente sem cruzar obstaculo, preenche o gap.
    Isso fecha ligacoes entre estradas paralelas proximas e juncoes incompletas.

    Prioridade: menor desvio angular em relacao a direcao natural da extremidade.
    """
    h, w = skeleton.shape
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    img_g   = img_rgb[:, :, 1].astype(np.int32)
    img_r   = img_rgb[:, :, 0].astype(np.int32)
    img_b   = img_rgb[:, :, 2].astype(np.int32)

    skel = (skeleton > 0).astype(np.uint8)
    k8   = np.ones((3, 3), np.uint8); k8[1, 1] = 0
    nbrs = cv2.filter2D(skel, -1, k8)
    endpoints = np.argwhere(skel & (nbrs == 1))

    result_map   = surface_map.copy()
    result_width = width_map.copy()
    road_mask    = (surface_map > 0).astype(np.uint8)

    # Varredura angular: do mais proximo da direcao de avanco ao mais distante
    SEARCH_ANGLES = [0, -20, 20, -40, 40, -60, 60, -80, 80, -100, 100, -120, 120, 180]

    for ep in endpoints:
        sy, sx = int(ep[0]), int(ep[1])
        if result_map[sy, sx] == 0:
            continue

        stype = int(surface_map[sy, sx])
        ref_w = max(float(width_map[sy, sx]), 2.0)
        r0, r1 = max(0, sy-4), min(h, sy+5)
        c0, c1 = max(0, sx-4), min(w, sx+5)
        ref_lab = img_lab[r0:r1, c0:c1].mean(axis=(0, 1))

        dir_vec  = _skeleton_direction(skel, sy, sx, look_back=12)
        base_dy, base_dx = dir_vec if dir_vec is not None else (0.0, 1.0)

        best_path  = None
        best_angle = float('inf')

        for angle_deg in SEARCH_ANGLES:
            rad = np.radians(float(angle_deg))
            ca, sa = np.cos(rad), np.sin(rad)
            tdy = base_dy * ca - base_dx * sa
            tdx = base_dy * sa + base_dx * ca

            path    = []
            blocked = False
            cy, cx  = float(sy), float(sx)

            for _step in range(max_gap + 1):
                cy += tdy
                cx += tdx
                tny = int(round(cy))
                tnx = int(round(cx))

                if not (0 <= tny < h and 0 <= tnx < w):
                    blocked = True
                    break

                if road_prob_map is not None and float(road_prob_map[tny, tnx]) < 0.001:
                    blocked = True
                    break

                if img_g[tny, tnx] - img_r[tny, tnx] > 22 and \
                   img_g[tny, tnx] - img_b[tny, tnx] > 18:
                    blocked = True
                    break

                cdist = float(np.linalg.norm(img_lab[tny, tnx] - ref_lab))
                
                local_color_tol = color_tol
                if road_prob_map is not None:
                    p = float(road_prob_map[tny, tnx])
                    if p > 0.25:
                        local_color_tol += 60
                        
                if cdist > local_color_tol:
                    blocked = True
                    break

                path.append((tny, tnx))

                # Chegou em estrada existente (diferente do ponto de origem)
                if road_mask[tny, tnx] > 0 and len(path) > 1:
                    abs_ang = abs(angle_deg)
                    if best_path is None or abs_ang < best_angle:
                        best_path  = path
                        best_angle = abs_ang
                    break

        if best_path:
            for py, px in best_path:
                if result_map[py, px] == 0:
                    result_map[py, px] = stype
                    result_width[py, px] = ref_w
                    road_mask[py, px]   = 1  # atualiza para proximas extremidades

    return result_map, result_width


def prune_dangling_stubs(surface_map, width_map, img_rgb,
                          road_prob_map=None,
                          max_gap_retry=70, color_tol_retry=70,
                          min_stub_px=30):
    """
    Etapa final de limpeza em duas fases:

    Fase 1 — tentativa extra de conexao (parametros relaxados):
      Para cada extremidade solta, varre ate max_gap_retry px com tolerancia
      de cor mais alta. Se encontrar estrada existente, conecta.

    Fase 2 — remocao de pontas soltas curtas:
      Recomputa esqueleto. Para cada extremidade, rastrea o ramo ate a proxima
      juncao (>=3 vizinhos) ou outra extremidade. Se o ramo tiver <=min_stub_px
      pixels de esqueleto, e claramente ruido e e removido do surface_map.
    """
    h, w = surface_map.shape
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    img_g   = img_rgb[:, :, 1].astype(np.int32)
    img_r   = img_rgb[:, :, 0].astype(np.int32)
    img_b   = img_rgb[:, :, 2].astype(np.int32)

    result_map   = surface_map.copy()
    result_width = width_map.copy()

    k8   = np.ones((3, 3), np.uint8); k8[1, 1] = 0
    DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    SEARCH_ANGLES = [0, -20, 20, -40, 40, -60, 60, -80, 80, -100, 100, -120, 120, 180]

    def _make_skel(smap):
        try:
            from skimage.morphology import skeletonize
            return skeletonize(smap > 0).astype(np.uint8)
        except ImportError:
            return (smap > 0).astype(np.uint8)

    # ------------------------------------------------------------------ Fase 1
    skel1      = _make_skel(result_map)
    nbrs1      = cv2.filter2D(skel1, -1, k8)
    endpoints1 = np.argwhere(skel1 & (nbrs1 == 1))
    road_mask  = (result_map > 0).astype(np.uint8)

    for ep in endpoints1:
        sy, sx = int(ep[0]), int(ep[1])
        if result_map[sy, sx] == 0:
            continue
        stype   = int(result_map[sy, sx])
        ref_w   = max(float(result_width[sy, sx]), 2.0)
        ref_lab = img_lab[max(0,sy-4):min(h,sy+5),
                          max(0,sx-4):min(w,sx+5)].mean(axis=(0, 1))
        dir_vec        = _skeleton_direction(skel1, sy, sx, look_back=12)
        base_dy, base_dx = dir_vec if dir_vec is not None else (0.0, 1.0)

        best_path  = None
        best_angle = float('inf')

        for angle_deg in SEARCH_ANGLES:
            rad = np.radians(float(angle_deg))
            ca, sa = np.cos(rad), np.sin(rad)
            tdy = base_dy * ca - base_dx * sa
            tdx = base_dy * sa + base_dx * ca
            path = []
            cy, cx = float(sy), float(sx)

            for _ in range(max_gap_retry + 1):
                cy += tdy;  cx += tdx
                tny = int(round(cy));  tnx = int(round(cx))
                if not (0 <= tny < h and 0 <= tnx < w):
                    break
                if road_prob_map is not None and float(road_prob_map[tny, tnx]) < 0.001:
                    break
                if img_g[tny,tnx]-img_r[tny,tnx]>22 and img_g[tny,tnx]-img_b[tny,tnx]>18:
                    break
                cdist = float(np.linalg.norm(img_lab[tny, tnx] - ref_lab))
                local_color_tol_retry = color_tol_retry
                if road_prob_map is not None:
                    p = float(road_prob_map[tny, tnx])
                    if p > 0.25:
                        local_color_tol_retry += 60
                if cdist > local_color_tol_retry:
                    break
                path.append((tny, tnx))
                if road_mask[tny, tnx] > 0 and len(path) > 1:
                    if best_path is None or abs(angle_deg) < best_angle:
                        best_path  = path
                        best_angle = abs(angle_deg)
                    break

        if best_path:
            for py, px in best_path:
                if result_map[py, px] == 0:
                    result_map[py, px] = stype
                    result_width[py, px] = ref_w
                    road_mask[py, px]   = 1

    # ------------------------------------------------------------------ Fase 2 (iterativa ate convergencia)
    # Remover um stub pode expor um novo stub na juncao que ficou com 2 ramos.
    # Repete ate que nenhum stub novo seja encontrado (max 8 rodadas).
    for _round in range(8):
        skel2      = _make_skel(result_map)
        nbrs2      = cv2.filter2D(skel2, -1, k8)
        junc2      = (skel2 > 0) & (nbrs2 >= 3)
        endpoints2 = np.argwhere(skel2 & (nbrs2 == 1))
        removed_now = 0

        for ep in endpoints2:
            sy, sx = int(ep[0]), int(ep[1])
            if result_map[sy, sx] == 0:
                continue

            branch  = [(sy, sx)]
            visited = {(sy, sx)}
            cy, cx  = sy, sx

            for _ in range(min_stub_px + 10):
                nxt = None
                for dy, dx in DIRS:
                    ny, nx = cy + dy, cx + dx
                    if not (0 <= ny < h and 0 <= nx < w): continue
                    if not skel2[ny, nx]:                  continue
                    if (ny, nx) in visited:                continue
                    nxt = (ny, nx); break
                if nxt is None:
                    break
                ny, nx = nxt
                if junc2[ny, nx]:
                    break  # chegou na juncao — parar sem incluir
                branch.append((ny, nx))
                visited.add((ny, nx))
                cy, cx = ny, nx

            if len(branch) <= min_stub_px:
                for py, px in branch:
                    r = max(1, min(int(round(float(result_width[py, px]) / 2)), 10))
                    ry0 = max(0, py - r);  ry1 = min(h, py + r + 1)
                    rx0 = max(0, px - r);  rx1 = min(w, px + r + 1)
                    result_map[ry0:ry1, rx0:rx1]   = 0
                    result_width[ry0:ry1, rx0:rx1] = 0
                removed_now += len(branch)

        if removed_now == 0:
            break  # convergiu

    return result_map, result_width


def draw_road_overlay(img_rgb, surface_map, width_map, skeleton=None):
    """
    Pinta cada componente da estrada com espessura uniforme, retangular e suave.
    Utiliza o esqueleto (skeleton) dilatado pela mediana da largura do componente,
    garantindo que conexões finas (pontes) herdem a largura padronizada da via a qual se conectam.
    Cores: asfalto=(50,50,50), terra=(160,90,30)
    """
    import math
    COLORS_RGB = {1: (50, 50, 50), 2: (160, 90, 30)}

    canvas   = np.zeros(img_rgb.shape, dtype=np.uint8)
    alpha    = np.zeros(img_rgb.shape[:2], dtype=np.float32)
    
    if skeleton is None:
        try:
            from skimage.morphology import skeletonize
            skel = skeletonize(surface_map > 0).astype(np.uint8)
        except ImportError:
            skel = (surface_map > 0).astype(np.uint8)
    else:
        skel = (skeleton > 0).astype(np.uint8)

    n_comps, comp_labels = cv2.connectedComponents(skel, connectivity=8)

    for lid in range(1, n_comps):
        comp_skel = (comp_labels == lid).astype(np.uint8)
        
        # Recuperar classes e larguras baseadas no esqueleto dilatado levemente
        # para pegar os valores do surface_map/width_map original
        probe = cv2.dilate(comp_skel, np.ones((3,3), np.uint8))
        types = surface_map[probe > 0]
        types = types[types > 0]
        if len(types) == 0:
            continue
            
        vals, cnts = np.unique(types, return_counts=True)
        stype = int(vals[np.argmax(cnts)])
        color = COLORS_RGB.get(stype, (50, 50, 50))

        widths = width_map[probe > 0]
        widths = widths[widths > 0]
        if len(widths) > 0:
            median_w = float(np.median(widths))
            if math.isnan(median_w) or median_w < 1:
                median_w = 4.0
        else:
            median_w = 4.0
            
        # Raio da dilatacao = metade da largura
        r = max(2, min(int(round(median_w / 2.0)), 16))
        # Para garantir "retangular uniforme, suave", usamos MORPH_ELLIPSE para curvas suaves
        k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        
        # Dilatar o esqueleto uniformemente
        dilated = cv2.dilate(comp_skel, k_d)
        
        dil_f = dilated.astype(np.float32)
        for c_idx in range(3):
            canvas[:, :, c_idx] = np.where(dilated > 0, color[c_idx], canvas[:, :, c_idx])
        alpha = np.maximum(alpha, dil_f)

    # Blur de borda para que as vias fiquem extremamente suaves ("suave, continua")
    canvas_blur = cv2.GaussianBlur(canvas, (0, 0), sigmaX=1.5)
    alpha_blur  = np.clip(cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.5), 0.0, 1.0)
    
    alpha3 = alpha_blur[:, :, np.newaxis] * 0.72
    result = img_rgb.astype(np.float32) * (1.0 - alpha3) + canvas_blur * alpha3
    return result.clip(0, 255).astype(np.uint8)


def build_road_graph(surface_map, skeleton):
    """
    Constroi grafo topologico a partir do esqueleto.
    - Vertices: juncoes (>=3 vizinhos) e pontas (1 vizinho), agrupados
    - Arestas: BFS no proprio esqueleto (parar ao atingir outro vertice)
      Isso garante que vias curvas e finas sejam conectadas corretamente.
    """
    h, w = skeleton.shape
    skel = (skeleton > 0).astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skel, -1, kernel)

    junctions = skel & (neighbor_count >= 3)
    endpoints = skel & (neighbor_count == 1)
    vertex_mask = (junctions | endpoints).astype(np.uint8)

    vy, vx = np.where(vertex_mask > 0)

    # Agrupar pixels de vertice proximos (raio 4px) num unico no
    groups = []       # (cx, cy, stype, pixel_set)
    used_px = set()

    for y, x in zip(vy.tolist(), vx.tolist()):
        if (y, x) in used_px:
            continue
        pixels = [(y, x)]
        used_px.add((y, x))
        q = deque([(y, x)])
        while q:
            cy, cx = q.popleft()
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < h and 0 <= nx < w
                            and (ny, nx) not in used_px
                            and vertex_mask[ny, nx] > 0):
                        used_px.add((ny, nx))
                        pixels.append((ny, nx))
                        q.append((ny, nx))

        cy_avg = sum(p[0] for p in pixels) / len(pixels)
        cx_avg = sum(p[1] for p in pixels) / len(pixels)
        types  = [surface_map[p[0], p[1]] for p in pixels
                  if surface_map[p[0], p[1]] > 0]
        stype  = max(set(types), key=types.count) if types else 1
        groups.append((cx_avg, cy_avg, stype, set(pixels)))

    if not groups:
        import torch
        return torch.zeros((0, 2)), [], []

    # Mapa pixel -> vertex id para acesso O(1) durante BFS
    pixel_to_vid = np.full((h, w), -1, dtype=np.int32)
    for vid, (cx, cy, stype, pixels) in enumerate(groups):
        for (py, px) in pixels:
            pixel_to_vid[py, px] = vid

    import torch
    nodes    = torch.tensor([(g[0], g[1]) for g in groups], dtype=torch.float32)
    vertices = [(g[0], g[1], g[2]) for g in groups]
    edges    = []
    edge_set = set()
    NAMES    = {1: "asfalto", 2: "terra", 3: "trilha"}

    MAX_TRAVEL = 600  # pixels maximos percorridos no esqueleto por BFS

    for start_vid, (sx, sy, stype_s, start_pixels) in enumerate(groups):
        # Iniciar BFS pelos vizinhos imediatos dos pixels do vertice
        queue   = deque()
        visited = set(start_pixels)

        for (py, px) in start_pixels:
            for dy, dx in [(-1,-1),(-1,0),(-1,1),
                           (0,-1),          (0,1),
                           (1,-1), (1,0), (1,1)]:
                ny, nx = py + dy, px + dx
                if (0 <= ny < h and 0 <= nx < w
                        and (ny, nx) not in visited
                        and skel[ny, nx] > 0):
                    visited.add((ny, nx))
                    queue.append((ny, nx, 1))

        while queue:
            cy, cx, dist = queue.popleft()
            if dist > MAX_TRAVEL:
                continue

            end_vid = int(pixel_to_vid[cy, cx])
            if end_vid >= 0 and end_vid != start_vid:
                ek = (min(start_vid, end_vid), max(start_vid, end_vid))
                if ek not in edge_set:
                    edge_set.add(ek)
                    stype_e = groups[end_vid][2]
                    st = stype_s if stype_s == stype_e else max(stype_s, stype_e)
                    edges.append((start_vid, end_vid, NAMES.get(st, "estrada"), 1.0))
                continue  # nao atravessar o vertice destino

            for dy, dx in [(-1,-1),(-1,0),(-1,1),
                           (0,-1),          (0,1),
                           (1,-1), (1,0), (1,1)]:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < h and 0 <= nx < w
                        and (ny, nx) not in visited
                        and skel[ny, nx] > 0):
                    visited.add((ny, nx))
                    queue.append((ny, nx, dist + 1))

    return nodes, edges, vertices
