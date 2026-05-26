"""
Pipeline leve para deteccao de vias — todas as larguras, rapido.
Sem trace direcional (passo mais caro). Usa classify.py para tudo.
"""

import os
import numpy as np
import cv2
import torch
from models.encoder import SAMEncoder
from models.ramo_b import GeometryDecoder


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class ThickRoadDetector:
    PATCH = 512

    def __init__(self, checkpoint="weights/cityscale_vitb_512_e10.ckpt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SAMEncoder(img_size=self.PATCH).to(self.device).eval()
        self.decoder = GeometryDecoder().to(self.device).eval()
        self._load(checkpoint)

    def _load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        sd   = ckpt["state_dict"]
        enc  = {k.replace("image_encoder.", ""): v for k, v in sd.items() if k.startswith("image_encoder.")}
        dec  = {k.replace("map_decoder.",   ""): v for k, v in sd.items() if k.startswith("map_decoder.")}
        self.encoder.encoder.load_state_dict(enc, strict=True)
        self.decoder.decoder.load_state_dict(dec, strict=True)
        print("Pesos carregados.")

    @torch.no_grad()
    def predict_at_scale(self, img_rgb, scale_factor=1.0):
        orig_h, orig_w = img_rgb.shape[:2]
        if abs(scale_factor - 1.0) > 1e-3:
            nw = int(orig_w * scale_factor)
            nh = int(orig_h * scale_factor)
            img_proc = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            img_proc = img_rgb

        ph, pw = img_proc.shape[:2]
        pad_h  = max(self.PATCH, ph)
        pad_w  = max(self.PATCH, pw)
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded[:ph, :pw] = img_proc

        stride   = self.PATCH // 2
        road_acc = np.zeros((pad_h, pad_w), dtype=np.float32)
        kp_acc   = np.zeros((pad_h, pad_w), dtype=np.float32)
        cnt_acc  = np.zeros((pad_h, pad_w), dtype=np.float32)

        positions = []
        y = 0
        while y + self.PATCH <= pad_h:
            x = 0
            while x + self.PATCH <= pad_w:
                positions.append((y, x))
                x += stride
            if pad_w > self.PATCH and (x - stride + self.PATCH) < pad_w:
                positions.append((y, pad_w - self.PATCH))
            y += stride
        if pad_h > self.PATCH and (y - stride + self.PATCH) < pad_h:
            x = 0
            while x + self.PATCH <= pad_w:
                positions.append((pad_h - self.PATCH, x))
                x += stride
        positions = list(set(positions))
        print(f"  {scale_factor:.1f}x: {pw}x{ph} -> {len(positions)} patches")

        batch_size = 8
        for i in range(0, len(positions), batch_size):
            batch_pos = positions[i : i+batch_size]
            
            tensors = []
            for (y0, x0) in batch_pos:
                patch  = padded[y0:y0+self.PATCH, x0:x0+self.PATCH].astype(np.float32)
                tensors.append(torch.from_numpy(patch).permute(2, 0, 1))
                
            batch_tensor = torch.stack(tensors).to(self.device)
            scores = torch.sigmoid(self.decoder(self.encoder(batch_tensor)))
            
            scores_np = scores.cpu().numpy()
            for b_idx, (y0, x0) in enumerate(batch_pos):
                road_acc[y0:y0+self.PATCH, x0:x0+self.PATCH] += scores_np[b_idx, 1]
                kp_acc  [y0:y0+self.PATCH, x0:x0+self.PATCH] += scores_np[b_idx, 0]
                cnt_acc [y0:y0+self.PATCH, x0:x0+self.PATCH] += 1.0

        valid = cnt_acc > 0
        road_acc[valid] /= cnt_acc[valid]
        kp_acc  [valid] /= cnt_acc[valid]

        road_map = road_acc[:ph, :pw]
        kp_map   = kp_acc  [:ph, :pw]

        if abs(scale_factor - 1.0) > 1e-3:
            road_map = cv2.resize(road_map, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
            kp_map   = cv2.resize(kp_map,   (orig_w, orig_h), interpolation=cv2.INTER_AREA)

        return road_map, kp_map


# ---------------------------------------------------------------------------
# Pre-processamento
# ---------------------------------------------------------------------------

def _remove_map_text(img_rgb):
    hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    cand = ((hsv[:, :, 2] > 205) & (hsv[:, :, 1] < 30)).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    mask = np.zeros_like(cand)
    for lid in range(1, n):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        cw   = int(stats[lid, cv2.CC_STAT_WIDTH])
        ch   = int(stats[lid, cv2.CC_STAT_HEIGHT])
        if not (4 <= area <= 500 and max(cw, ch) < 90):
            continue
        comp = (lbl == lid).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            rw, rh = cv2.minAreaRect(max(cnts, key=cv2.contourArea))[1]
            if min(rw, rh) > 0.5 and max(rw, rh) / min(rw, rh) >= 3.5:
                continue
        mask[lbl == lid] = 1
    if mask.sum() == 0:
        return img_rgb
    k = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, k, iterations=2)
    img_rgb[mask > 0] = [0, 200, 0]
    return img_rgb


def _hysteresis_binary(prob_map, low, high):
    """
    Hysteresis: pixels >= high sao sementes; pixels >= low so entram se
    conectados a uma semente. Reconstroi trechos esmaecidos no meio de vias
    com pontas confiantes. Fallback manual se skimage nao tiver.
    """
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


def _filter_blob_shapes(binary, typical_width):
    """
    Remove componentes nao-elongados (telhados, manchas): area/bbox_area > 0.55
    em areas pequenas a medias. Estradas tem densidade <0.3 dentro do bbox.
    """
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = binary.copy()
    max_blob_area = int(typical_width * typical_width * 6.0)
    for lid in range(1, n):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        bw   = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh   = int(stats[lid, cv2.CC_STAT_HEIGHT])
        bbox = max(bw * bh, 1)
        density = area / bbox
        # Telhado: denso (>0.55) E pequeno/medio (< 6*w²). Rua: densidade baixa.
        if density > 0.55 and area < max_blob_area:
            result[lbl == lid] = 0
    return result


def _draw_from_prob_map(img_rgb, fused_road, surface_map, typical_width):
    """
    Reconstroi overlay direto do mapa de probabilidade:
      1. Hysteresis (low/high) → binaria com trechos fracos preservados se
         conectados a sementes confiantes (resolve descontinuidades).
      2. Fechamento morfologico → fecha gaps curtos
      3. Filtro de linearidade → remove blobs (telhados, manchas)
      4. Esqueletizacao
      5. Filtro de comprimento minimo
      6. Dilatacao uniforme + blur de borda
    """
    road_max = float(fused_road.max())
    if road_max < 1e-4:
        return img_rgb.copy()

    # 1. Hysteresis: low captura meio esmaecido, high garante semente confiavel
    low_thr  = max(road_max * 0.12, 0.0015)
    high_thr = max(road_max * 0.45, 0.006)
    binary = _hysteresis_binary(fused_road, low_thr, high_thr)

    # 2. Fechar gaps curtos
    r_c = max(2, min(int(typical_width * 0.5), 12))
    k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_c+1, 2*r_c+1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_c)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # 3. Remover blobs (telhados, manchas nao-elongadas)
    binary = _filter_blob_shapes(binary, typical_width)

    # 4. Esqueletizar
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(binary > 0).astype(np.uint8)
    except ImportError:
        skel = binary.copy()

    # 5. Remover stubs curtos: comprimento < 3 * typical_width
    min_len = max(20, int(typical_width * 3.0))
    n_s, s_labels = cv2.connectedComponents(skel, connectivity=8)
    skel_clean = np.zeros_like(skel)
    for lid in range(1, n_s):
        if int((s_labels == lid).sum()) >= min_len:
            skel_clean[s_labels == lid] = 1

    # 5. Dilatar cada componente uniformemente
    r_d = max(2, min(int(round(typical_width / 2)), 16))
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_d+1, 2*r_d+1))

    COLORS = {1: np.array([50,  50,  50 ], np.float32),
              2: np.array([160, 90,  30 ], np.float32)}

    n_c, c_labels = cv2.connectedComponents(skel_clean, connectivity=8)
    canvas = np.zeros(img_rgb.shape, np.float32)
    alpha  = np.zeros(img_rgb.shape[:2], np.float32)

    for lid in range(1, n_c):
        comp = (c_labels == lid).astype(np.uint8)
        dil  = cv2.dilate(comp, k_d)

        types = surface_map[dil > 0]
        types = types[types > 0]
        if len(types) > 0:
            vals, cnts = np.unique(types, return_counts=True)
            stype = int(vals[np.argmax(cnts)])
        else:
            stype = 1
        color = COLORS.get(stype, COLORS[1])

        dil_f = dil.astype(np.float32)
        for c in range(3):
            canvas[:, :, c] = np.where(dil > 0, color[c], canvas[:, :, c])
        alpha = np.maximum(alpha, dil_f)

    # 6. Blur de borda
    canvas_blur = cv2.GaussianBlur(canvas, (0, 0), sigmaX=2.5)
    alpha_blur  = np.clip(cv2.GaussianBlur(alpha, (0, 0), sigmaX=3.0), 0.0, 1.0)

    alpha3 = alpha_blur[:, :, np.newaxis] * 0.72
    result = img_rgb.astype(np.float32) * (1.0 - alpha3) + canvas_blur * alpha3
    return result.clip(0, 255).astype(np.uint8)


def _make_skel(surface_map):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(surface_map > 0).astype(np.uint8) * 255
    except ImportError:
        img  = (surface_map > 0).astype(np.uint8) * 255
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = np.zeros_like(img)
        tmp  = img.copy()
        for _ in range(100):
            eroded = cv2.erode(tmp, elem)
            temp   = cv2.subtract(tmp, cv2.dilate(eroded, elem))
            skel   = cv2.bitwise_or(skel, temp)
            tmp    = eroded.copy()
            if cv2.countNonZero(tmp) == 0:
                break
        return skel


# ---------------------------------------------------------------------------
# Binarizacao sensiblizada: aceita sinais fracos e preenche gaps sob arvores
# ---------------------------------------------------------------------------

def _binarize_sensitive(fused_road, fused_keypoint, typical_width):
    """
    Versao mais sensiveldo enhance_road_mask:
    - thr_strong = road_max * 0.15  (vs 0.20 no pipeline completo)
    - thr_weak   = road_max * 0.02
    - Dilata strong_bin antes de rotular para fechar gaps sob copas de arvores
    """
    road_max   = float(fused_road.max())
    thr_strong = float(np.clip(road_max * 0.15, 0.004, 0.025))
    thr_weak   = float(np.clip(road_max * 0.02, 0.0008, 0.005))

    strong_bin = (fused_road > thr_strong).astype(np.uint8)
    if fused_keypoint is not None:
        strong_bin = np.maximum(strong_bin,
                                (fused_keypoint > 0.04).astype(np.uint8))

    # Dilatar strong_bin pelo raio da via: gaps curtos (copa de arvore, sombra)
    # entre dois trechos fortes ficam no mesmo componente → detectados como ponte
    gap_r        = max(5, min(int(typical_width * 1.5), 28))
    strong_grown = cv2.dilate(strong_bin,
                              np.ones((2*gap_r+1, 2*gap_r+1), np.uint8))

    full_bin = (fused_road > thr_weak).astype(np.uint8)

    _, grown_labels  = cv2.connectedComponents(strong_grown, connectivity=8)
    n_full, full_labels, full_stats, _ = cv2.connectedComponentsWithStats(
        full_bin, connectivity=8)

    result    = np.zeros_like(strong_bin, dtype=np.uint8)
    min_area  = max(4, int(typical_width * 0.4))
    max_blob  = int(typical_width * typical_width * 20)

    for fid in range(1, n_full):
        comp_area = int(full_stats[fid, cv2.CC_STAT_AREA])
        if comp_area < min_area:
            continue
        comp      = full_labels == fid
        sl_vals   = grown_labels[comp & (strong_grown > 0)]
        unique_sl = set(sl_vals.tolist()); unique_sl.discard(0)

        if len(unique_sl) >= 2:
            bw = int(full_stats[fid, cv2.CC_STAT_WIDTH])
            bh = int(full_stats[fid, cv2.CC_STAT_HEIGHT])
            if comp_area > max_blob and max(bw,bh)/max(min(bw,bh),1) < 1.5:
                result[comp & (strong_bin > 0)] = 255
            else:
                result[comp] = 255
        else:
            result[comp & (strong_bin > 0)] = 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(result, cv2.MORPH_OPEN, k3)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(image_path, checkpoint="weights/cityscale_vitb_512_e10.ckpt"):
    from pipeline.classify import (
        measure_road_widths,
        classify_road_surface,
        unify_segment_surface,
        filter_non_road_shapes,
        close_road_gaps,
        prune_dangling_stubs,
    )

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Imagem nao encontrada: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    print(f"Imagem: {w}x{h}")

    # img_rgb  = _remove_map_text(img_rgb)
    # DESATIVADO para evitar borrar carros e telhados claros
    detector = ThickRoadDetector(checkpoint)

    # --- Escala Scout (0.5x) ---
    print("Rodando escala scout 0.5x para imagens de alto zoom...")
    road_05x, kp_05x = detector.predict_at_scale(img_rgb, 0.5)

    # Estimar largura tipica
    _probe = (road_05x > 0.015).astype(np.uint8) * 255
    _dist  = cv2.distanceTransform(_probe, cv2.DIST_L2, 5)
    _rpx   = _probe > 0
    typical_width = (float(np.percentile(_dist[_rpx], 75)) * 2.0
                     if _rpx.sum() > 300 else 8.0)
    typical_width = float(np.clip(typical_width, 4.0, 60.0))
    print(f"Largura tipica: {typical_width:.1f}px (via scout)")

    if typical_width > 16.0:
        print("Zoom alto detectado — usando escalas 0.5x e 1.0x...")
        road_1, kp_1 = road_05x, kp_05x
        road_2, kp_2 = detector.predict_at_scale(img_rgb, 1.0)
    else:
        print("Zoom padrao detectado — usando escalas 1.0x e 2.0x...")
        road_1, kp_1 = detector.predict_at_scale(img_rgb, 1.0)
        road_2, kp_2 = detector.predict_at_scale(img_rgb, 2.0)
        
    fused_road     = np.maximum(road_1, road_2)
    fused_keypoint = np.maximum(kp_1, kp_2)

    # --- Binarizacao e Bridge Gaps (Exato como no prompt) ---
    from pipeline.bridge_gaps import run as run_bridge_gaps
    print("Rodando Bridge Gaps (Hysteresis, EE, EL, Dijkstra)...")
    road_binary = run_bridge_gaps(fused_road)
    print(f"Road binaria: {(road_binary > 0).sum()}px")

    # --- Parametros adaptativos (mais leves que o pipeline completo) ---
    u = typical_width
    p_max_gap   = max(30, min(int(u * 6.0), 150))   # generoso: gaps de copa de arvore
    p_min_area  = max(20, min(int(u * u * 0.4), 400))
    p_gap_retry = max(60, min(int(u * 7.0), 180))
    p_min_stub  = max(12, min(int(u * 2.0), 55))

    # --- Classificacao ---
    width_map   = measure_road_widths(road_binary)
    surface_map = classify_road_surface(img_rgb, road_binary, width_map)
    surface_map = unify_segment_surface(surface_map)
    surface_map = filter_non_road_shapes(surface_map, min_aspect=2.5,
                                         min_area=p_min_area)
    print(f"Apos classificacao: {(surface_map > 0).sum()}px")

    # --- Fechar gaps e podar pontas (pipeline leve: sem trace direcional) ---
    skel_tmp = _make_skel(surface_map)
    # road_prob_map=None: deixa cor + cheque de vegetacao guiarem o gap-filling
    # em vez do road_prob (que e 0 sob copas de arvores urbanas).
    surface_map, width_map = close_road_gaps(
        surface_map, width_map, skel_tmp, img_rgb,
        road_prob_map=None, max_gap=p_max_gap, color_tol=80
    )
    surface_map = unify_segment_surface(surface_map)

    surface_map, width_map = prune_dangling_stubs(
        surface_map, width_map, img_rgb,
        road_prob_map=None,
        max_gap_retry=p_gap_retry, color_tol_retry=85,
        min_stub_px=p_min_stub
    )
    surface_map = unify_segment_surface(surface_map)
    print(f"Final: {(surface_map > 0).sum()}px")

    # --- Overlay: reconstrucao direta do mapa de probabilidade limpo ---
    overlay = _draw_from_prob_map(img_rgb, fused_road, surface_map, typical_width)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/thick_overlay.png",
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite("outputs/thick_road_mask.png",
                (fused_road * 255).clip(0, 255).astype(np.uint8))
    surf_viz = np.zeros((*surface_map.shape, 3), dtype=np.uint8)
    surf_viz[surface_map == 1] = [40, 40, 40]
    surf_viz[surface_map == 2] = [160, 90, 30]
    cv2.imwrite("outputs/thick_surface.png",
                cv2.cvtColor(surf_viz, cv2.COLOR_RGB2BGR))

    print("Salvo em outputs/thick_overlay.png")
    return surface_map, overlay
