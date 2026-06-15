"""
Pre-processamento de entrada para a REDE NEURAL (versao final, enxuta).

Dos filtros classicos avaliados (gaussian, clahe, sharpen, sobel_fusion,
canny_fusion), o canny_fusion foi o de melhor resultado em QA e e o padrao
do pipeline: realca as bordas paralelas das vias misturando 15% do mapa de
Canny na imagem.

IMPORTANTE: a imagem pre-processada alimenta APENAS o modelo
(img_for_model). A classificacao de pavimento, as mascaras de contexto e o
overlay usam sempre a imagem ORIGINAL — bordas sinteticas do Canny nao
podem contaminar as evidencias de cor/textura.
"""

import cv2

_VALID_METHODS = {"none", "canny_fusion"}


def preprocess_image(img_rgb, method="canny_fusion"):
    """Retorna a imagem para o modelo. method: 'canny_fusion' (padrao) ou 'none'."""
    if method is None:
        method = "none"
    method = str(method).lower()
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Metodo invalido: {method}. Opcoes: {', '.join(sorted(_VALID_METHODS))}")

    if method == "none":
        return img_rgb

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(img_rgb, 0.85, edges_rgb, 0.15, 0)
