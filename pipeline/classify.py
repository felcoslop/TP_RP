"""
Evidencias de pavimento e medicao de largura (versao final, enxuta).

Mantem apenas as duas funcoes usadas pelo pipeline de grafo:
  - measure_road_widths       : largura por pixel via Distance Transform
  - compute_p_terra_evidence  : evidencia pixel-a-pixel de "terra" [0,1]

A decisao asfalto x terra acontece POR ARESTA em pipeline/graph_refine.py
(percentil 35 ao longo da via + suavizacao ICM + nos de transicao).
"""

import cv2
import numpy as np


def measure_road_widths(road_mask_binary):
    """
    Usa Distance Transform para medir a largura real de cada pixel de estrada.
    O valor em cada pixel = distancia ate a borda mais proxima * 2 = largura
    total da via naquele ponto (exato na centerline).
    """
    dist = cv2.distanceTransform(road_mask_binary, cv2.DIST_L2, 5)
    return dist * 2


def compute_p_terra_evidence(img_rgb):
    """
    Evidencia pixel-a-pixel de "terra" em [0,1] para a classificacao POR
    ARESTA. Combina:

      P_cor     : matiz quente HSV (hue ~14 +-12) * saturacao — terra e
                  marrom/laranja/amarelada; penalizada em pixels brancos
                  (asfalto claro/concreto) e cinza-neutros (asfalto).
      P_textura : desvio padrao local 7x7 do grayscale — terra e granular,
                  asfalto e liso.

    Notas de calibracao (validadas em QA):
      - SEM blur de vizinhanca: a agregacao espacial e feita ao longo da
        propria aresta do grafo (nao vaza entre vias paralelas).
      - white_penalty atenuado moderadamente pela textura (fator 0.4):
        terra clara/arenosa e brilhante MAS granular; o fator e moderado
        porque asfalto claro com carros/faixas tambem e brilhante e
        texturizado e nao pode virar terra.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue = img_hsv[:, :, 0].astype(np.float32)
    sat = img_hsv[:, :, 1].astype(np.float32)
    val = img_hsv[:, :, 2].astype(np.float32)

    warm_hue = np.clip(1.0 - np.abs(hue - 14.0) / 12.0, 0.0, 1.0)
    sat_prob = np.clip((sat - 25.0) / 100.0, 0.0, 1.0)
    p_color = (warm_hue * sat_prob).astype(np.float32)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean_l = cv2.boxFilter(img_gray, cv2.CV_32F, (7, 7))
    mean_sq = cv2.boxFilter(img_gray * img_gray, cv2.CV_32F, (7, 7))
    std_map = np.sqrt(np.maximum(mean_sq - mean_l * mean_l, 0.0))
    p_texture = np.clip((std_map - 8.0) / 22.0, 0.0, 1.0).astype(np.float32)

    white_penalty = (np.clip((val - 130.0) / 80.0, 0.0, 1.0) *
                     np.clip(1.0 - sat / 60.0, 0.0, 1.0)).astype(np.float32)
    white_penalty = white_penalty * (1.0 - 0.4 * p_texture)

    p_color = p_color * (1.0 - 0.90 * white_penalty)
    gray_road = (np.clip(1.0 - sat / 40.0, 0.0, 1.0) *
                 np.clip((val - 100.0) / 100.0, 0.0, 1.0)).astype(np.float32)
    p_color = p_color * (1.0 - 0.85 * gray_road)

    p_texture = p_texture * (1.0 - 0.70 * white_penalty)
    return (0.55 * p_color + 0.45 * p_texture).astype(np.float32)
