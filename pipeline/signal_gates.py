"""
Gates de SINAL que rodam ANTES da binarizacao.

Recebem o mapa de probabilidade do modelo e o suprimem (multiplicam por peso
< 1) em pixels visualmente incompativeis com asfalto/terra:

  - vegetacao verde dominante (copa de arvore, mato, gramado)
  - telhado saturado (vermelho/marrom telha, azul piscina, telhado verde
    saturado de zinco)

A supressao e GRADIENTE (nao hard-kill), entao se o modelo tinha confianca
muito alta numa rua sob copa de arvore (acontece em estrada visivel entre
arvores), uma parte do sinal sobrevive. Mas confianca media-baixa que
"vazaria" para a copa e fortemente reduzida.
"""

import cv2
import numpy as np


def vegetation_weight(img_rgb, soft=1.0):
    """
    Retorna mapa [0..1] onde 1 = pixel sem vegetacao, 0 = vegetacao densa.
    Usado para multiplicar road_prob.

    soft: quanto da supressao aplicar. 1.0 = total, 0.5 = parcial.
    """
    g = img_rgb[:, :, 1].astype(np.float32)
    r = img_rgb[:, :, 0].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)

    # Score crescente com a dominancia de verde
    g_dom = np.maximum(g - np.maximum(r, b), 0.0)
    # Tambem requerer saturacao razoavel — folhagem real, nao cinza-esverdeado
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    veg_score = np.clip(g_dom / 22.0, 0.0, 1.0) * np.clip(sat / 60.0, 0.0, 1.0)
    # peso = 1 - veg_score * soft
    return (1.0 - veg_score * soft).astype(np.float32)


def saturated_roof_weight(img_rgb, soft=0.85):
    """
    Penaliza pixels com saturacao muito alta tipica de telhado de telha
    (vermelho-alaranjado) ou telhado pintado. Asfalto tem sat baixa.

    Retorna [0..1].
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_ = hsv[:, :, 0].astype(np.float32)
    s_ = hsv[:, :, 1].astype(np.float32)
    v_ = hsv[:, :, 2].astype(np.float32)

    # Telha vermelha/marrom: hue 0-18 ou 165-179, sat>60, val 60-200
    red_hue = (h_ <= 18) | (h_ >= 165)
    brown   = red_hue & (s_ > 60) & (v_ > 55) & (v_ < 200)

    # Telhado azul (piscina, telha azul): hue 100-130, sat>80
    blue = (h_ >= 100) & (h_ <= 130) & (s_ > 80)

    # Score: 1 onde telhado, 0 onde nao
    score = np.zeros_like(h_, dtype=np.float32)
    score[brown] = 1.0
    score[blue]  = 1.0

    # Suavizar para nao criar buracos pixelados (telha tem variacao)
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=1.5)
    return (1.0 - score * soft).astype(np.float32)


def apply_signal_gates(road_prob, img_rgb,
                       veg_soft=0.85, roof_soft=0.70):
    """
    Aplica gates de vegetacao e telhado ao mapa de probabilidade.

    veg_soft=0.85: pixels totalmente verdes ficam com 15% da prob original.
                   Pixels mistos (rua sob arvore esparsa) sobrevivem bem.
    roof_soft=0.70: telhado vermelho fica com 30% da prob original.
                    Asfalto cinza/marrom de terra nao sao afetados.
    """
    w_veg = vegetation_weight(img_rgb, soft=veg_soft)
    w_rof = saturated_roof_weight(img_rgb, soft=roof_soft)
    return road_prob * w_veg * w_rof


def post_classify_green_filter(surface_map, img_rgb,
                                max_green_frac=0.35):
    """
    Apos a classificacao, varre cada COMPONENTE conectado. Se >max_green_frac
    dos pixels do componente sao verde-dominantes, o componente eh removido.
    Para componentes muito grandes (rede inteira), examina por SEGMENTO entre
    juncoes em vez de inteiro.
    """
    if surface_map.sum() == 0:
        return surface_map

    g = img_rgb[:, :, 1].astype(np.int32)
    r = img_rgb[:, :, 0].astype(np.int32)
    b = img_rgb[:, :, 2].astype(np.int32)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.int32)
    green_pix = ((g - r > 12) & (g - b > 8) & (sat > 45))

    rb = (surface_map > 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(rb, connectivity=8)
    out = surface_map.copy()
    for cid in range(1, n):
        comp = (lbl == cid)
        a = int(comp.sum())
        if a < 30:
            continue
        gfrac = float(green_pix[comp].sum()) / a
        # Componentes pequenos sao removidos se forem majoritariamente verdes
        if a < 600 and gfrac >= max_green_frac:
            out[comp] = 0
        # Componentes grandes ficam, mas pixels verdes individuais saem
        elif a >= 600:
            out[comp & green_pix] = 0
    return out
