"""
Motor principal do pipeline de deteccao de malha viaria (versao final).

Fluxo (uma unica via de processamento — a melhor validada):

  1. Inferencia multi-escala com SCOUT: uma passada rapida em 0.5x estima a
     largura tipica das vias e decide as escalas. Zoom alto (vias largas)
     processa 0.5x + 1.0x; zoom padrao/distante processa 1.0x + 2.0x — o
     UPSCALE 2x e o que permite enxergar estradas de terra muito finas.
     Fusao por maximo entre escalas. TTA opcional de rotacao (0/90/180/270).
  2. Mascaras de contexto (pipeline/context_filter.py): vegetacao, agua,
     telhados/predios e solo exposto suprimem a probabilidade ANTES da
     binarizacao (pixels de alta confianca nunca sao suprimidos).
  3. Binarizacao por hysteresis com limiares ABSOLUTOS (0.04/0.30) e
     fallback relativo para dominio de sinal fraco; guard-rail "sem vias".
  4. GRAFO como fonte unica de verdade (pipeline/graph_refine.py):
     esqueleto -> nos/arestas; fusao de nos; poda de pontas (inclusive as
     que morrem em telhado); PONTES por caminho de menor custo (Dijkstra
     direcional) sobre 1/(prob+eps) + contexto.
  5. Classificacao de pavimento POR ARESTA (percentil 35 da evidencia
     cor+textura no nucleo da via) + suavizacao ICM + nos de transicao
     quando a troca asfalto/terra e real. Terra = LARANJA, asfalto = AZUL.
  6. Overlay desenhado do grafo + export JSON/GraphML.

A imagem pre-processada (ex.: canny_fusion) alimenta APENAS a rede neural
(img_for_model); classificacao, mascaras e overlay usam a imagem original.
"""

import os
from datetime import datetime
import numpy as np
import cv2
import torch
from models.encoder import SAMEncoder
from models.ramo_b import GeometryDecoder

# Limiares absolutos de hysteresis (calibrados no operating point do
# SAM-Road, road~0.34).
HYST_LOW = 0.04
HYST_HIGH = 0.30
# Regime de sinal fraco (dominio fora do treino, ex.: satelite escuro com o
# ckpt CityScale): abaixo de HYST_HIGH caimos para limiares relativos
# 0.45/0.12 do maximo, desde que exista um minimo de sinal (WEAK_FLOOR) —
# imagens sem vias continuam vazias.
WEAK_FLOOR = 0.10

# Imagem verde (rural/floresta): se a vegetacao detectada cobre mais que esta
# fracao da imagem, o refino do grafo (pontes/podas/filtro de quarteirao,
# calibrado p/ malha URBANA) atrapalha e chega a remover ruas reais — paramos
# no GRAFO BRUTO e vamos direto ao overlay. Vegetacao e o sinal limpo de
# "natural" (telhado nao e verde). 0.25 separa rural de urbano com folga nas
# imagens de teste (urbano <=18.5% vs rural >=28.4%); na duvida, pular o refino
# e o lado seguro (grafo bruto raramente piora; o refino pode comer rua real).
VEG_SKIP_REFINE = 0.25

# Limiar de terra ADAPTATIVO por dominio (via fracao de vegetacao):
#   urbano (pouco verde) -> TERRA_THR alto (0.45): evita asfalto fino virar
#                           laranja;
#   rural/verde           -> TERRA_THR baixo (0.25): pega estrada de terra
#                           (terra seca clara fica em p_terra ~0.26, abaixo de
#                           0.45 mas acima de 0.25).
# Interpola linearmente entre as duas ancoras de vegetacao.
TERRA_THR_URBANO = 0.45
TERRA_THR_RURAL = 0.20
VEG_TERRA_LO = 0.10   # <= isso: urbano puro
VEG_TERRA_HI = 0.28   # >= isso: rural


def _adaptive_terra_thr(veg_frac):
    t = float(np.clip((veg_frac - VEG_TERRA_LO) / (VEG_TERRA_HI - VEG_TERRA_LO),
                      0.0, 1.0))
    return TERRA_THR_URBANO - (TERRA_THR_URBANO - TERRA_THR_RURAL) * t


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class ThickRoadDetector:
    """Encoder SAM ViT-B + Geometry Decoder do SAM-Road.

    patch=512 para o checkpoint cityscale_vitb_512;
    patch=256 para o checkpoint spacenet_vitb_256 e para o fine-tuned do Colab.
    """

    def __init__(self, checkpoint, patch=512):
        self.PATCH = int(patch)
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
        """Janela deslizante (stride = PATCH/2) em uma escala; retorna
        (prob_via, prob_keypoint) no tamanho original."""
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


def _predict_scale(detector, img_rgb, scale, tta=False):
    """Inferencia em uma escala, com TTA opcional de rotacao (0/90/180/270)."""
    if not tta:
        return detector.predict_at_scale(img_rgb, scale)
    roads, kps = [], []
    for k in range(4):
        img_r = np.ascontiguousarray(np.rot90(img_rgb, k))
        r, kp = detector.predict_at_scale(img_r, scale)
        roads.append(np.ascontiguousarray(np.rot90(r, -k)))
        kps.append(np.ascontiguousarray(np.rot90(kp, -k)))
    return np.mean(roads, axis=0), np.mean(kps, axis=0)


def predict_fused(detector, img_rgb, tta=False):
    """Scout 0.5x -> escolha automatica de escalas conforme o zoom ->
    fusao por maximo. E aqui que vias muito finas (zoom distante ou
    estradas de terra estreitas) ganham o UPSCALE 2x.
    Retorna (fused_road, fused_keypoint, typical_width)."""
    print("Rodando escala scout 0.5x...")
    road_05x, kp_05x = detector.predict_at_scale(img_rgb, 0.5)

    _probe = (road_05x > 0.015).astype(np.uint8) * 255
    _dist  = cv2.distanceTransform(_probe, cv2.DIST_L2, 5)
    _rpx   = _probe > 0
    typical_width = (float(np.percentile(_dist[_rpx], 75)) * 2.0
                     if _rpx.sum() > 300 else 8.0)
    typical_width = float(np.clip(typical_width, 4.0, 60.0))
    print(f"Largura tipica: {typical_width:.1f}px (via scout)")

    if typical_width > 16.0:
        print("Zoom alto detectado — usando escalas 0.5x e 1.0x...")
        if tta:
            road_1, kp_1 = _predict_scale(detector, img_rgb, 0.5, tta=True)
        else:
            road_1, kp_1 = road_05x, kp_05x
        road_2, kp_2 = _predict_scale(detector, img_rgb, 1.0, tta)
    else:
        print("Zoom padrao/distante detectado — usando escalas 1.0x e 2.0x...")
        road_1, kp_1 = _predict_scale(detector, img_rgb, 1.0, tta)
        road_2, kp_2 = _predict_scale(detector, img_rgb, 2.0, tta)

    fused_road     = np.maximum(road_1, road_2)
    fused_keypoint = np.maximum(kp_1, kp_2)
    return fused_road, fused_keypoint, typical_width


# ---------------------------------------------------------------------------
# Binarizacao
# ---------------------------------------------------------------------------

def _hysteresis_binary(prob_map, low, high):
    """Pixels >= high sao sementes; pixels >= low entram se conectados a uma
    semente — reconstroi trechos esmaecidos no meio de vias confiantes."""
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


# ---------------------------------------------------------------------------
# Visualizacoes de depuracao (1 imagem por passo do pipeline)
# ---------------------------------------------------------------------------

def _heatmap_on(img_rgb, prob):
    """Sobrepoe a probabilidade de via (mapa de calor) numa copia escurecida."""
    pm = np.clip(prob / max(float(prob.max()), 1e-6), 0, 1)
    cmap = cv2.applyColorMap((pm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    base = (img_rgb.astype(np.float32) * 0.35).astype(np.uint8)
    a = (pm[:, :, None] * 0.9)
    return (base * (1 - a) + cmap * a).clip(0, 255).astype(np.uint8)


def _binary_on(img_rgb, binary):
    """Pinta a mascara binaria (ciano) sobre a imagem escurecida."""
    out = (img_rgb.astype(np.float32) * 0.45).astype(np.uint8)
    out[binary > 0] = (0, 255, 255)
    return out


def _draw_graph_debug(img_rgb, G, edge_color=(0, 255, 255)):
    """Desenha arestas (linhas) + nos (juncao=vermelho, ponta=amarelo) para
    inspecionar a topologia, antes da classificacao de pavimento."""
    out = (img_rgb.astype(np.float32) * 0.45).astype(np.uint8)
    for e in G.edges.values():
        pts = np.array([(p[1], p[0]) for p in e["path"]], np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, edge_color, 2, cv2.LINE_AA)
    for nid, nd in G.nodes.items():
        c = (int(round(nd["pos"][1])), int(round(nd["pos"][0])))
        deg = G.degree(nid)
        col = (255, 60, 60) if deg >= 3 else ((255, 230, 0) if deg == 1 else (0, 200, 255))
        cv2.circle(out, c, 4, col, -1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(img_rgb, detector, use_context=True, tta=False,
                 img_for_model=None, debug_steps=False, vias_claras="auto",
                 bright_vmin=205, road_thr=HYST_HIGH, toco_frac=0.80):
    """Nucleo reutilizavel (tambem usado por eval/).

    img_rgb       : imagem ORIGINAL — mascaras de contexto, classificacao e
                    overlay usam sempre esta.
    img_for_model : versao pre-processada (canny_fusion) vista APENAS pela
                    rede neural. Se None, usa img_rgb.
    debug_steps   : se True, result["steps"] recebe 1 imagem por etapa do
                    pipeline (para entender/visualizar a classificacao).
    vias_claras   : "auto" (padrao) suprime vias claras so em zoom proximo/
                    urbano; "nao" sempre suprime (modo urbano); "sim" nunca
                    suprime (satelite alto / rural com estradas de terra).
    Retorna dict: graph, surface_map, overlay, prob, masks, stats, fused_road.
    """
    from pipeline.context_filter import (build_context_masks,
                                         apply_suppression,
                                         masks_debug_image)
    from pipeline.graph_refine import (graph_from_mask, refine_graph,
                                       classify_edges,
                                       split_surface_transitions,
                                       icm_smooth, rasterize_surface,
                                       draw_graph_overlay, graph_stats,
                                       RoadGraph)
    from pipeline.classify import measure_road_widths, compute_p_terra_evidence

    if img_for_model is None:
        img_for_model = img_rgb

    fused_road, fused_keypoint, typical_width = predict_fused(
        detector, img_for_model, tta=tta)

    masks = None
    prob = fused_road
    if use_context:
        # Portao de brilho: zoom proximo/urbano (largura tipica grande) ->
        # rua de verdade e asfalto ESCURO; suprime linhas claras/neutras.
        if vias_claras == "nao":
            reject_bright = True
        elif vias_claras == "sim":
            reject_bright = False
        else:  # auto, pela largura tipica estimada pelo scout
            reject_bright = typical_width >= 14.0
        # Densidade da malha viaria (para separar rural esparso de favela densa)
        road_density = float((fused_road > 0.30).mean())
        print("Construindo mascaras de contexto "
              f"(vegetacao/agua/telhado/solo | vias_claras={vias_claras}"
              f" -> rejeita_claras={reject_bright})...")
        masks = build_context_masks(img_rgb, typical_width,
                                    road_prob=fused_road,
                                    reject_bright=reject_bright,
                                    bright_vmin=bright_vmin,
                                    road_density=road_density)
        prob = apply_suppression(fused_road, masks)
    else:
        road_density = 1.0

    result = {
        "masks": masks,
        "prob": prob,
        "fused_road": fused_road,
        "typical_width": typical_width,
        "masks_viz": None,   # construido apos a binarizacao (recortando as ruas)
    }

    steps = {} if debug_steps else None
    if debug_steps:
        steps["1_original"] = img_rgb.copy()
        if img_for_model is not img_rgb:
            steps["2_preprocessamento_canny"] = img_for_model.copy()
        steps["3_probabilidade_modelo"] = _heatmap_on(img_rgb, fused_road)
        if masks is not None:
            steps["5_probabilidade_suprimida"] = _heatmap_on(img_rgb, prob)
            # passo 4 (mascaras) e gerado apos a binarizacao, recortando ruas

    # Limiar em dois niveis: absoluto (regime normal) com fallback relativo
    # para dominio de sinal fraco; guard-rail abaixo de WEAK_FLOOR.
    # road_thr e a TOLERANCIA: na mascara de via, so vira rua o branco >= road_thr.
    # Subir road_thr corta as linhas FRACAS (vias falsas). O piso da hysteresis
    # sobe junto (max(HYST_LOW, road_thr-0.26)) para nao crescer de volta o fraco.
    hi_abs = float(road_thr)
    lo_abs = max(HYST_LOW, hi_abs - 0.26)
    road_max = float(prob.max())
    if road_max >= hi_abs:
        hyst_low, hyst_high = lo_abs, hi_abs
    elif road_max >= WEAK_FLOOR:
        hyst_high = road_max * 0.45
        hyst_low = road_max * 0.12
        print(f"Sinal fraco (max={road_max:.3f}) — limiares relativos "
              f"{hyst_low:.3f}/{hyst_high:.3f}")
    else:
        print("Nenhuma via com confianca suficiente detectada nesta imagem.")
        if masks is not None:
            result["masks_viz"] = masks_debug_image(img_rgb, masks)
        G = RoadGraph(img_rgb.shape[:2])
        result.update({
            "graph": G,
            "surface_map": np.zeros(img_rgb.shape[:2], np.uint8),
            "overlay": img_rgb.copy(),
            "stats": graph_stats(G),
        })
        return result

    binary = _hysteresis_binary(prob, hyst_low, hyst_high)
    r_c = max(2, min(int(typical_width * 0.5), 6))
    k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_c + 1, 2 * r_c + 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_c)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    print(f"Binaria (hysteresis {hyst_low:.3f}/{hyst_high:.3f}): "
          f"{(binary > 0).sum()}px")

    # Recorta as ruas DETECTADAS da mascara de telhado/bloco: o vermelho nao
    # aparece mais sobre rua de verdade (no viz) e o filtro de bloco nao confunde
    # rua com quarteirao. Usa a binaria final (nao so a probabilidade).
    if masks is not None and masks["roof"].any():
        rdil = cv2.dilate(binary, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * max(2, int(typical_width)) + 1,) * 2))
        masks["roof"][rdil > 0] = 0
    if masks is not None:
        result["masks_viz"] = masks_debug_image(img_rgb, masks)
        if debug_steps:
            steps["4_mascaras_contexto"] = result["masks_viz"]
    if debug_steps:
        steps["6_binaria"] = _binary_on(img_rgb, binary)

    width_map = measure_road_widths(binary)
    G = graph_from_mask(binary, width_map)
    print(f"Grafo bruto: {len(G.nodes)} nos, {len(G.edges)} arestas")
    if debug_steps:
        steps["7_grafo_bruto"] = _draw_graph_debug(img_rgb, G)

    # RURAL = muito verde E malha esparsa (favela densa em vale verde NAO e
    # rural). Em rural pula o refino (calibrado p/ malha urbana, distorce no
    # campo) e leva o grafo bruto direto ao overlay.
    from pipeline.context_filter import is_rural as _is_rural
    veg_frac = float(masks["vegetation"].mean()) if masks is not None else 0.0
    rural = _is_rural(veg_frac, road_density)
    if rural:
        print(f"RURAL (verde {veg_frac*100:.0f}%, malha {road_density*100:.2f}%) "
              f"— pulando refino; grafo bruto direto para o overlay.")
        refine_stats = {"refino_pulado_rural": True,
                        "veg_frac": round(veg_frac, 3),
                        "road_density": round(road_density, 4)}
    else:
        # IMPORTANTE: o Dijkstra de pontes usa a probabilidade BRUTA do modelo
        # (fused_road) — a supressao de contexto vale so para a binarizacao.
        # Vias sob copas de arvores continuam conectaveis pelo sinal residual.
        refine_stats = refine_graph(G, fused_road, masks, typical_width,
                                    toco_frac=toco_frac)
    print(f"Refino: {refine_stats}")
    if debug_steps:
        steps["8_grafo_refinado"] = _draw_graph_debug(img_rgb, G)

    # Limiar de terra por DOMINIO (rural vs urbano, decidido por verde + malha):
    # rural -> baixo (pega terra); urbano/favela -> alto (mais asfalto, evita
    # asfalto fino/largo virar laranja).
    terra_thr = TERRA_THR_RURAL if rural else TERRA_THR_URBANO
    print(f"Limiar de terra: {terra_thr:.2f} ({'rural' if rural else 'urbano'})")
    p_terra = compute_p_terra_evidence(img_rgb)
    classify_edges(G, p_terra, thr=terra_thr)
    n_splits = split_surface_transitions(G, thr=terra_thr)
    n_flips = icm_smooth(G)
    from pipeline.graph_refine import flip_isolated_terra
    n_iso = flip_isolated_terra(G)
    print(f"Superficie por aresta: {n_splits} transicoes reais, "
          f"{n_flips} arestas suavizadas (ICM), {n_iso} terra isolada->asfalto")

    surface_map = rasterize_surface(G, img_rgb.shape[:2])
    overlay = draw_graph_overlay(img_rgb, G)
    stats = graph_stats(G)
    stats.update({f"refine_{k}": v for k, v in refine_stats.items()})
    print(f"Grafo final: {stats}")

    if debug_steps:
        steps["9_overlay_final"] = overlay

    result.update({
        "graph": G,
        "surface_map": surface_map,
        "overlay": overlay,
        "stats": stats,
        "steps": steps,
    })
    return result


def run(image_path, checkpoint, img_rgb=None, use_context=True, tta=False,
        export_graph=True, out_dir="outputs", img_for_model=None, patch=512,
        debug_steps=False, vias_claras="auto", bright_vmin=205,
        road_thr=HYST_HIGH, toco_frac=0.80):
    """Executa o pipeline completo e grava as saidas em out_dir.
    Retorna (surface_map, overlay)."""
    from pipeline.export import save_graph_json, save_graph_graphml
    from pipeline.graph_refine import COLORS_RGB

    if img_rgb is None:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Imagem nao encontrada: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"Imagem: {w}x{h} | contexto: {use_context} | tta: {tta} | patch: {patch}")

    detector = ThickRoadDetector(checkpoint, patch=patch)
    result = run_pipeline(img_rgb, detector, use_context=use_context,
                          tta=tta, img_for_model=img_for_model,
                          debug_steps=debug_steps, vias_claras=vias_claras,
                          bright_vmin=bright_vmin, road_thr=road_thr,
                          toco_frac=toco_frac)

    # Cada execucao vai para a SUA PROPRIA subpasta com data/hora. Assim a
    # pasta mais recente e obvia, nada e sobrescrito, e os nomes de arquivo
    # ficam limpos (sem confundir qual e o mais novo).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    overlay = result["overlay"]
    surface_map = result["surface_map"]

    def _out(name):
        return os.path.join(run_dir, name)

    cv2.imwrite(_out("thick_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(_out("thick_road_mask.png"),
                (result["fused_road"] * 255).clip(0, 255).astype(np.uint8))

    surf_viz = np.zeros((*surface_map.shape, 3), dtype=np.uint8)
    surf_viz[surface_map == 1] = COLORS_RGB[1]
    surf_viz[surface_map == 2] = COLORS_RGB[2]
    cv2.imwrite(_out("thick_surface.png"),
                cv2.cvtColor(surf_viz, cv2.COLOR_RGB2BGR))

    if result.get("masks_viz") is not None:
        cv2.imwrite(_out("context_masks.png"),
                    cv2.cvtColor(result["masks_viz"], cv2.COLOR_RGB2BGR))

    if export_graph and result.get("graph") is not None:
        src = os.path.basename(image_path) if image_path else None
        save_graph_json(result["graph"], _out("graph_output.json"),
                        source_image=src)
        save_graph_graphml(result["graph"], _out("graph_output.graphml"))

    # Imagens de cada passo do pipeline (--passos): subpasta passos/
    if result.get("steps"):
        passos_dir = os.path.join(run_dir, "passos")
        os.makedirs(passos_dir, exist_ok=True)
        for nome, im in result["steps"].items():
            cv2.imwrite(os.path.join(passos_dir, f"{nome}.png"),
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print(f">>> Passos do pipeline em: {passos_dir}")

    print(f"\n>>> Resultado desta execucao na pasta: {run_dir}")
    print(f">>> Abra: {os.path.join(run_dir, 'thick_overlay.png')}")
    return surface_map, overlay
