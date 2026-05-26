"""
Pre-processador opcional: detecta vegetacao, telhados saturados e areas
muito escuras (sombras profundas) e as substitui por um plano de fundo
cinza neutro borrado. A ideia e remover do campo de visao do modelo
pixels que claramente NAO sao rua, reduzindo falsos positivos.

ATENCAO — TRADEOFF:
  - Ajuda em imagens com muita vegetacao confundindo o modelo
  - PODE prejudicar recall em ruas finas SOB copa de arvore (sao apagadas)

Por isso e opcional, atras do flag --remove-areas no run_fast.py.

Heuristicas combinadas:
  - Cor: HSV + dominancia RGB
  - Textura: desvio padrao local (granularidade)
  - Forma: dilata e remove componentes pequenos para nao apagar pixels
    isolados que possam ser de rua
"""

import cv2
import numpy as np


# -------------------------------------------------------------------------
# Detectores individuais
# -------------------------------------------------------------------------

def _vegetation_mask(img_rgb, min_area=300):
    """
    Vegetacao: verde dominante + sat razoavel + agrupamento espacial.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    g = img_rgb[:, :, 1].astype(np.int32)
    r = img_rgb[:, :, 0].astype(np.int32)
    b = img_rgb[:, :, 2].astype(np.int32)
    sat = hsv[:, :, 1]

    green_dom = ((g - r) > 12) & ((g - b) > 8)
    veg = (green_dom & (sat > 50)).astype(np.uint8)

    # Suavizar e agrupar
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, k, iterations=2)
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, k, iterations=1)

    # So manter regioes grandes (vegetacao agrupada, nao pixel isolado
    # que pode ser de rua)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(veg, connectivity=8)
    keep = np.zeros_like(veg)
    for lid in range(1, n):
        if int(stats[lid, cv2.CC_STAT_AREA]) >= min_area:
            keep[lbl == lid] = 1
    return keep


def _saturated_roof_mask(img_rgb, min_area=120):
    """
    Telhado de telha vermelho/marrom (hue 0-18 ou 165-179, sat alta),
    azul (telhado pintado ou piscina), ou amarelo forte.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_ = hsv[:, :, 0]
    s_ = hsv[:, :, 1]
    v_ = hsv[:, :, 2]

    red_hue   = (h_ <= 18) | (h_ >= 165)
    brown_roof = red_hue & (s_ > 70) & (v_ > 60) & (v_ < 210)
    blue_roof  = (h_ >= 100) & (h_ <= 130) & (s_ > 80)
    yellow_roof = (h_ >= 22) & (h_ <= 35) & (s_ > 110) & (v_ > 120)

    roof = (brown_roof | blue_roof | yellow_roof).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    roof = cv2.morphologyEx(roof, cv2.MORPH_CLOSE, k, iterations=2)
    roof = cv2.morphologyEx(roof, cv2.MORPH_OPEN, k, iterations=1)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(roof, connectivity=8)
    keep = np.zeros_like(roof)
    for lid in range(1, n):
        if int(stats[lid, cv2.CC_STAT_AREA]) >= min_area:
            keep[lbl == lid] = 1
    return keep


def _bright_metal_roof_mask(img_rgb, min_area=200):
    """
    Telhado de zinco/aluminio claro: muito brilhante (val>210), saturacao
    bem baixa. Mas asfalto desgastado e faixas tambem sao brancos —
    para evitar apagar faixas, exigir AREA grande (faixa eh fina).
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s_ = hsv[:, :, 1]
    v_ = hsv[:, :, 2]
    bright = ((v_ > 210) & (s_ < 35)).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k, iterations=1)

    # Filtro de tamanho + razao de aspecto: telhado e relativamente "redondo",
    # faixa e MUITO alongada. Manter so o "redondo".
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    keep = np.zeros_like(bright)
    for lid in range(1, n):
        a = int(stats[lid, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        cw = int(stats[lid, cv2.CC_STAT_WIDTH])
        ch = int(stats[lid, cv2.CC_STAT_HEIGHT])
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if aspect > 6.0:
            continue  # provavelmente faixa, nao telhado
        keep[lbl == lid] = 1
    return keep


def _deep_shadow_mask(img_rgb, min_area=400):
    """
    Sombra profunda: val muito baixo (V<55) + sat baixa, em regioes grandes.
    Asfalto pode estar nessa faixa mas raramente eh tao baixo (val<55).
    Conservador: so pega o muito escuro.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v_ = hsv[:, :, 2]
    s_ = hsv[:, :, 1]
    dark = ((v_ < 55) & (s_ < 60)).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k, iterations=1)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros_like(dark)
    for lid in range(1, n):
        if int(stats[lid, cv2.CC_STAT_AREA]) >= min_area:
            keep[lbl == lid] = 1
    return keep


def _high_texture_mask(img_rgb, std_threshold=22, min_area=400):
    """
    Areas de textura muito alta: copas de arvore (granularidade) e telhados
    com padroes repetitivos. Asfalto/terra sao relativamente lisos.

    Usa desvio padrao local 9x9.
    """
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m1 = cv2.boxFilter(g, cv2.CV_32F, (9, 9))
    m2 = cv2.boxFilter(g * g, cv2.CV_32F, (9, 9))
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    hi = (std > std_threshold).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hi = cv2.morphologyEx(hi, cv2.MORPH_OPEN, k, iterations=1)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(hi, connectivity=8)
    keep = np.zeros_like(hi)
    for lid in range(1, n):
        if int(stats[lid, cv2.CC_STAT_AREA]) >= min_area:
            keep[lbl == lid] = 1
    return keep


# -------------------------------------------------------------------------
# Combinacao e aplicacao
# -------------------------------------------------------------------------

def build_cleanup_mask(img_rgb, include_shadows=False,
                       require_double_evidence_for_texture=True):
    """
    Constroi mascara combinada de areas a serem suprimidas.

    Regras de seguranca:
      - Telhado metalico claro: so se forma nao for muito alongada
      - Textura alta: so se TAMBEM tiver cor de vegetacao OU telhado
        (evita apagar paving texturizado)
      - Sombras: opcional, default desligado
    """
    veg   = _vegetation_mask(img_rgb)
    roof  = _saturated_roof_mask(img_rgb)
    metal = _bright_metal_roof_mask(img_rgb)
    texture = _high_texture_mask(img_rgb)

    if require_double_evidence_for_texture:
        # Textura so vira mascara se sobrepuser cor de vegetacao/telhado
        # (i.e., reforca evidencia, nao adiciona pixels novos)
        texture = texture & (veg | roof | metal)

    full = (veg | roof | metal | texture).astype(np.uint8)

    if include_shadows:
        full = full | _deep_shadow_mask(img_rgb)
        full = full.astype(np.uint8)

    # Dilatar levemente para alcancar bordas
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    full = cv2.dilate(full, k, iterations=1)

    return {
        "full":    full,
        "veg":     veg,
        "roof":    roof,
        "metal":   metal,
        "texture": texture,
    }


def apply_cleanup(img_rgb, mask_dict, blur_sigma=8.0,
                  neutral_color=(140, 140, 140), neutral_blend=0.55):
    """
    Aplica a mascara a imagem:
      1. Gaussian blur forte na imagem inteira
      2. Onde a mascara esta ativa, mistura o blur com cor neutra
         (asfalto desgastado, cinza-bege). Isso garante que o modelo
         "veja" um plano de fundo sem feicoes de rua nem feicoes que
         excitem falsos positivos.

    neutral_blend=0.55 -> 55% cinza neutro + 45% blur da imagem.
    """
    h, w = img_rgb.shape[:2]
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=blur_sigma)

    neutral = np.full_like(img_rgb, neutral_color, dtype=np.uint8)
    blended = (blurred.astype(np.float32) * (1.0 - neutral_blend) +
               neutral.astype(np.float32) * neutral_blend).astype(np.uint8)

    m = (mask_dict["full"] > 0)
    out = img_rgb.copy()
    out[m] = blended[m]

    # Suavizar transicao nas bordas da mascara para nao criar arestas duras
    soft = cv2.GaussianBlur(mask_dict["full"].astype(np.float32),
                            (0, 0), sigmaX=4.0)
    soft = np.clip(soft, 0.0, 1.0)[:, :, np.newaxis]
    out = (img_rgb.astype(np.float32) * (1.0 - soft) +
           out.astype(np.float32) * soft).clip(0, 255).astype(np.uint8)

    return out


def clean_scene(img_rgb, include_shadows=False, blur_sigma=8.0,
                 neutral_color=(140, 140, 140), neutral_blend=0.55):
    """Entrypoint de conveniencia."""
    md = build_cleanup_mask(img_rgb, include_shadows=include_shadows)
    return apply_cleanup(img_rgb, md, blur_sigma=blur_sigma,
                         neutral_color=neutral_color,
                         neutral_blend=neutral_blend), md
