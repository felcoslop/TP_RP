"""
Detecção de bordas baseada na tese de doutorado de Braga (2018):
"Navegação Autônoma de VANT por Imagens LiDAR" — INPE.

Implementa os filtros de remoção de ruído (Mediana e Gaussiano) e os
detectores de borda (Canny, Sobel, MLP-LUT) descritos no Capítulo 6.

Modos disponíveis (3 detectores × 2 filtros + 2 extras):
  fast         — Canny com auto-thresholds + Mediana 3×3  (iteração rápida)
  canny_gauss  — Canny + Gaussiano 5×5  (Seção 6.6.2.1)
  canny_median — Canny + Mediana 3×3    (Seção 6.6.2.4)
  sobel_gauss  — Sobel + Gaussiano 5×5  (Seção 6.6.2.2)
  sobel_median — Sobel + Mediana 3×3    (Seção 6.6.2.5)
  mlp_gauss    — MLP-LUT + Gaussiano    (Seção 6.6.2.3)
  mlp_median   — MLP-LUT + Mediana      (Seção 6.6.2.6)
  combined     — Fusão Canny+Sobel (max) + Mediana  (máximo recall)
  none         — Desativado (pipeline original sem bordas)

Referência:
  Braga, J.R.G. (2018). Navegação Autônoma de VANT por Imagens LiDAR.
  Tese de Doutorado, INPE. sid.inpe.br/mtc-m21c/2018/05.18.16.04-TDI
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

VALID_MODES = [
    "fast", "canny_gauss", "canny_median",
    "sobel_gauss", "sobel_median",
    "mlp_gauss", "mlp_median",
    "combined", "none",
]

# Peso da evidência de bordas na fusão com road_prob do modelo ViT.
# 70% modelo + 30% bordas  (ajustável via parâmetro).
DEFAULT_EDGE_WEIGHT = 0.30


# ---------------------------------------------------------------------------
# 1. Filtros de remoção de ruído  (Seção 6.2)
# ---------------------------------------------------------------------------

def apply_noise_filter(img_gray, method="median", ksize=3, sigma=1.0):
    """
    Aplica filtro de remoção de ruído em imagem em tons de cinza.

    Parâmetros
    ----------
    img_gray : ndarray (H, W), uint8
        Imagem monocromática.
    method : str
        'median'   — Filtro da Mediana (Seção 6.2.1)
        'gaussian' — Filtro Gaussiano  (Seção 6.2.2)
    ksize : int
        Tamanho do kernel (deve ser ímpar). Default 3 para mediana, 5 para
        gaussiano.
    sigma : float
        Desvio padrão do kernel gaussiano (ignorado para mediana).

    Retorna
    -------
    ndarray (H, W), uint8
        Imagem filtrada.
    """
    if method == "median":
        return cv2.medianBlur(img_gray, ksize)
    elif method == "gaussian":
        return cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)
    else:
        raise ValueError(f"Filtro desconhecido: {method}")


# ---------------------------------------------------------------------------
# 2. Detectores de borda  (Seção 6.4)
# ---------------------------------------------------------------------------

def detect_edges_canny(img_gray, low=None, high=None, auto=False):
    """
    Operador de Canny (Seção 6.4.3).

    Detecção multi-etapa: suavização gaussiana → gradientes (Sobel interno) →
    supressão não-máxima → histerese com limiar duplo.

    Se auto=True, calcula thresholds adaptativos pela mediana de Otsu.

    Retorna mapa binário uint8 (0 ou 255).
    """
    if auto or (low is None and high is None):
        # Threshold adaptativo via mediana da intensidade
        v = float(np.median(img_gray))
        low = int(max(0, 0.50 * v))
        high = int(min(255, 1.30 * v))
        # Garantir mínimos razoáveis
        low = max(low, 20)
        high = max(high, 50)
    else:
        low = low or 50
        high = high or 150

    return cv2.Canny(img_gray, low, high)


def detect_edges_sobel(img_gray, ksize=3, threshold=None):
    """
    Operador de Sobel (Seção 6.4.2).

    Convolução com kernels 3×3 para gradientes Gx e Gy,
    seguida de cálculo da magnitude: G = sqrt(Gx² + Gy²).

    Retorna mapa de bordas uint8 (0 ou 255) binarizado via Otsu se
    threshold=None, ou pelo threshold fornecido.
    """
    # Gradientes em x e y
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    # Magnitude
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # Normalizar para 0-255
    mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8) \
        if mag.max() > 0 else np.zeros_like(img_gray)

    if threshold is None:
        # Binarização automática por Otsu (método clássico)
        _, binary = cv2.threshold(mag, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
    return binary


def detect_edges_mlp_lut(img_gray, window=5, threshold=None):
    """
    Aproximação da MLP em LUT (Seção 6.4.4).

    A tese treina uma MLP (Multilayer Perceptron) para classificar janelas
    5×5 como borda ou não-borda, implementada em FPGA como Look-Up Table.

    Sem os pesos originais, implementamos uma aproximação que captura o
    comportamento descrito: análise de gradiente multi-direcional em janela
    5×5 com classificação borda/não-borda baseada na variância local e
    magnitude do gradiente.

    O método combina:
    - Gradiente Scharr (mais preciso que Sobel para ângulos oblíquos)
    - Variância local em janela 5×5 (captura textura de borda)
    - Gradiente direcional (45° e 135°) para bordas diagonais

    Retorna mapa binário uint8 (0 ou 255).
    """
    h, w = img_gray.shape
    img_f = img_gray.astype(np.float32)

    # --- Componente 1: Gradiente Scharr (mais preciso que Sobel 3×3) ---
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    mag_scharr = np.sqrt(gx ** 2 + gy ** 2)

    # --- Componente 2: Gradientes diagonais (kernels 3×3) ---
    k45 = np.array([[-1, -1, 0],
                     [-1,  0, 1],
                     [ 0,  1, 1]], dtype=np.float32)
    k135 = np.array([[ 0, -1, -1],
                      [ 1,  0, -1],
                      [ 1,  1,  0]], dtype=np.float32)
    g45 = np.abs(cv2.filter2D(img_f, cv2.CV_32F, k45))
    g135 = np.abs(cv2.filter2D(img_f, cv2.CV_32F, k135))

    # --- Componente 3: Variância local em janela 5×5 ---
    # Na tese, a MLP analisa padrões 5×5. A variância local captura
    # a "textura de borda" que a rede neural aprenderia.
    mean_l = cv2.boxFilter(img_f, cv2.CV_32F, (window, window))
    mean_sq = cv2.boxFilter(img_f * img_f, cv2.CV_32F, (window, window))
    var_local = np.maximum(mean_sq - mean_l * mean_l, 0.0)
    std_local = np.sqrt(var_local)

    # --- Combinação ponderada (simula os pesos da MLP) ---
    # Normalizar cada componente para [0, 1]
    def _norm(x):
        mx = x.max()
        return x / mx if mx > 0 else x

    score = (0.45 * _norm(mag_scharr) +
             0.20 * _norm(np.maximum(g45, g135)) +
             0.35 * _norm(std_local))

    # Normalizar para 0-255
    score_u8 = np.clip(score / score.max() * 255, 0, 255).astype(np.uint8) \
        if score.max() > 0 else np.zeros_like(img_gray)

    if threshold is None:
        _, binary = cv2.threshold(score_u8, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(score_u8, threshold, 255,
                                  cv2.THRESH_BINARY)
    return binary


# ---------------------------------------------------------------------------
# 3. Pipeline de alto nível: modo → máscara de bordas
# ---------------------------------------------------------------------------

def get_edge_mask(img_rgb, mode="fast"):
    """
    Retorna máscara binária de bordas (0/255) para a imagem RGB dada.

    Aplica a combinação filtro de ruído + detector de borda conforme o
    modo selecionado (ver docstring do módulo).

    Parâmetros
    ----------
    img_rgb : ndarray (H, W, 3), uint8
        Imagem original em RGB.
    mode : str
        Um dos VALID_MODES.

    Retorna
    -------
    edge_mask : ndarray (H, W), uint8  (valores 0 ou 255)
    """
    if mode == "none":
        return np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    if mode not in VALID_MODES:
        raise ValueError(
            f"Modo inválido: {mode}. Válidos: {VALID_MODES}")

    # Converter para tons de cinza (Seção 6.1 — Equação 6.4)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # --- Selecionar filtro e detector com base no modo ---
    if mode == "fast":
        filtered = apply_noise_filter(gray, "median", ksize=3)
        edges = detect_edges_canny(filtered, auto=True)

    elif mode == "canny_gauss":
        filtered = apply_noise_filter(gray, "gaussian", ksize=5, sigma=1.0)
        edges = detect_edges_canny(filtered, auto=True)

    elif mode == "canny_median":
        filtered = apply_noise_filter(gray, "median", ksize=3)
        edges = detect_edges_canny(filtered, auto=True)

    elif mode == "sobel_gauss":
        filtered = apply_noise_filter(gray, "gaussian", ksize=5, sigma=1.0)
        edges = detect_edges_sobel(filtered)

    elif mode == "sobel_median":
        filtered = apply_noise_filter(gray, "median", ksize=3)
        edges = detect_edges_sobel(filtered)

    elif mode == "mlp_gauss":
        filtered = apply_noise_filter(gray, "gaussian", ksize=5, sigma=1.0)
        edges = detect_edges_mlp_lut(filtered)

    elif mode == "mlp_median":
        filtered = apply_noise_filter(gray, "median", ksize=3)
        edges = detect_edges_mlp_lut(filtered)

    elif mode == "combined":
        # Fusão Canny + Sobel para máximo recall
        filtered = apply_noise_filter(gray, "median", ksize=3)
        e_canny = detect_edges_canny(filtered, auto=True)
        e_sobel = detect_edges_sobel(filtered)
        edges = np.maximum(e_canny, e_sobel)

    return edges


# ---------------------------------------------------------------------------
# 4. Integração com o pipeline de detecção de estradas
# ---------------------------------------------------------------------------

def edge_enhance_road_prob(road_prob, img_rgb, mode="fast",
                           edge_weight=DEFAULT_EDGE_WEIGHT,
                           dilate_radius=3):
    """
    Refina o mapa de probabilidade de estradas usando evidência de bordas.

    A lógica segue o princípio da tese: bordas estruturais na imagem
    confirmam a presença de vias. O refinamento opera em dois sentidos:

    1. **Boost**: Pixels com road_prob > limiar mínimo E que estão próximos
       de bordas detectadas recebem amplificação proporcional.

    2. **Supressão suave**: Áreas com road_prob fraco (< limiar) E sem
       bordas próximas recebem penalização suave (reduz falsos positivos
       em telhados e vegetação que não possuem bordas lineares).

    Parâmetros
    ----------
    road_prob : ndarray (H, W), float32
        Probabilidade de estrada do modelo ViT (0.0 a 1.0).
    img_rgb : ndarray (H, W, 3), uint8
        Imagem original em RGB.
    mode : str
        Modo de detecção de bordas.
    edge_weight : float
        Peso da evidência de bordas (0.0 = sem efeito, 1.0 = só bordas).
    dilate_radius : int
        Raio de dilatação das bordas (em pixels) para criar zona de
        influência ao redor de cada borda detectada.

    Retorna
    -------
    enhanced_prob : ndarray (H, W), float32
        Mapa de probabilidade refinado.
    edge_mask : ndarray (H, W), uint8
        Máscara de bordas gerada (para debug/visualização).
    """
    if mode == "none":
        return road_prob.copy(), np.zeros(road_prob.shape, dtype=np.uint8)

    # Gerar máscara de bordas
    edge_mask = get_edge_mask(img_rgb, mode)

    # Normalizar bordas para [0, 1] float
    edge_f = (edge_mask > 0).astype(np.float32)

    # Dilatar bordas para criar zona de influência
    # Bordas de estrada raramente são linhas perfeitas de 1px na imagem
    if dilate_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_radius + 1, 2 * dilate_radius + 1))
        edge_dilated = cv2.dilate(edge_f, k).astype(np.float32)
    else:
        edge_dilated = edge_f

    # Suavizar a zona de influência para transição gradual
    edge_field = cv2.GaussianBlur(edge_dilated, (0, 0), sigmaX=2.0)
    edge_field = np.clip(edge_field, 0.0, 1.0)

    # --- Refinamento ---
    rp_max = float(road_prob.max())
    if rp_max < 1e-6:
        return road_prob.copy(), edge_mask

    # 1. Boost Aditivo: Onde há bordas, adicionamos um valor fixo à probabilidade
    # para ajudar o algoritmo de rastreamento (trace_roads/stub_extension)
    # a não parar antes do fim da via, mesmo que a IA tenha dado prob quase 0.
    additive_boost = edge_field * edge_weight * 0.10

    # 2. Boost Multiplicativo: Amplificamos a probabilidade existente onde há bordas
    boost = edge_field * edge_weight
    enhanced = road_prob * (1.0 + boost) + additive_boost

    # NOTA: Supressão global removida a pedido do usuário.
    # O objetivo das bordas é COMPLEMENTAR (ajudar a chegar ao fim da via),
    # e não apagar estradas reais que não têm bordas nítidas.

    # Clamp
    enhanced = np.clip(enhanced, 0.0, rp_max * 1.5)

    return enhanced.astype(np.float32), edge_mask
