"""
Mascaras de contexto (Fase 1 do plano v2).

Constroi mascaras de exclusao a partir da imagem RGB para suprimir falsos
positivos ANTES da binarizacao do mapa de probabilidade:

  - vegetation : florestas, parques, copas densas (indice ExG + HSV)
  - water      : rios, lagos, mar (baixa textura + matiz fria + area grande)
  - roof       : telhados e construcoes (cor quente saturada OU branco
                 brilhante, em componente compacto/retangular)
  - soil       : solo exposto / desmatamento (paleta de terra, mas forma 2-D
                 — discriminado de estrada de terra pela elongacao do
                 esqueleto: estrada e estrutura 1-D)

As mesmas mascaras sao reutilizadas como custo/obstaculo no A* de
pontes (pipeline/graph_refine.py).

Regras de supressao (multiplicador sobre o mapa de probabilidade):
  vegetation, water, roof -> 0.0 (supressao total)
  soil                    -> 0.3 (parcial: estrada de terra real pode cruzar)
Pixels com probabilidade alta (>= protect_thr) nunca sao suprimidos, para
nao apagar vias em que o modelo tem confianca (ex.: ponte sobre rio).
"""

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
except ImportError:  # fallback degrade: trata tudo como 1-D (nao remove)
    skeletonize = None

SOIL_KEEP = 0.3          # fator residual de prob. em solo exposto
ROOF_KEEP = 0.25         # fator residual de prob. em telhado/bloco (supressao suave)
PROTECT_THR = 0.45       # prob. acima da qual o pixel nunca e suprimido
# Imagem RURAL = MUITO VERDE **E** MALHA VIARIA ESPARSA. A vegetacao sozinha nao
# basta: uma favela num vale verde tem 40%+ de verde mas malha DENSA (e urbana).
# A densidade da via (fracao de pixels com prob > 0.30) separa limpo: rural de
# verdade <= 0.6%; favela/urbano >= 2%. Em rural, telhado e solo nao se aplicam
# (so pintariam campos de vermelho e suprimiriam estrada de terra fina).
ROOF_SKIP_VEG = 0.25
ROAD_DENSITY_RURAL = 0.012   # malha abaixo disso (+ muito verde) = rural


def _local_std(gray_f32, k=7):
    mean = cv2.boxFilter(gray_f32, cv2.CV_32F, (k, k))
    mean_sq = cv2.boxFilter(gray_f32 * gray_f32, cv2.CV_32F, (k, k))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _vegetation_mask(img_rgb, hsv):
    r = img_rgb[:, :, 0].astype(np.int16)
    g = img_rgb[:, :, 1].astype(np.int16)
    b = img_rgb[:, :, 2].astype(np.int16)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Correcao de cast global: imagens de satelite bruto podem ter veu verde
    # na cena inteira (ex.: SpaceNet Xangai), disparando a dominancia em 100%
    # dos pixels. O vies e estimado nos pixels NEUTROS (baixa saturacao —
    # ruas/telhados cinza), nao na cena toda, para nao cancelar florestas
    # reais em imagens majoritariamente verdes.
    neutral = sat < 60
    if float(neutral.mean()) > 0.02:
        bias_gr = float(np.clip(np.median((g - r)[neutral]), -25, 25))
        bias_gb = float(np.clip(np.median((g - b)[neutral]), -25, 25))
    else:
        bias_gr = bias_gb = 0.0

    # Verde dominante (regra do stub_smart, com vies de cast) OU matiz verde.
    green_dom = ((g - r) > 12 + bias_gr) & ((g - b) > 8 + bias_gb)
    green_hue = (hue >= 35) & (hue <= 85) & (sat > 40) & (val > 25)
    # Vegetacao CLARA/SECA/amarronzada (mato seco, capim, folhagem clara): matiz
    # amarelo-verde (28-50) com verde >= azul e g nao menor que r. O limite
    # inferior 28 fica ACIMA do solo vermelho/dirt (matiz ~5-22), entao nao
    # captura solo exposto nem estrada de terra.
    dry_green = ((hue >= 28) & (hue <= 50) & (sat > 25) & (val > 35) &
                 (g >= b) & (g >= r - 6))
    veg = (green_dom | green_hue | dry_green).astype(np.uint8)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, k5)   # remove pontinhos
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, k7)  # consolida copas
    return veg


def _water_mask(img_rgb, hsv, std_map):
    h, w = img_rgb.shape[:2]
    r = img_rgb[:, :, 0].astype(np.int16)
    g = img_rgb[:, :, 1].astype(np.int16)
    b = img_rgb[:, :, 2].astype(np.int16)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    low_texture = std_map < 7.0
    bluish = (b - r >= 8) & (b >= g)
    # exige azul >= verde para nao capturar copa de floresta escura/hazy
    dark_cyan = (hue >= 90) & (hue <= 135) & (sat > 20) & (val < 150) & (b >= g)
    cand = (low_texture & (bluish | dark_cyan)).astype(np.uint8)

    cand = cv2.morphologyEx(
        cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # Agua = componente grande e homogeneo; evita confundir asfalto azulado
    min_area = max(3000, int(0.004 * h * w))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    water = np.zeros((h, w), np.uint8)
    for lid in range(1, n):
        if int(stats[lid, cv2.CC_STAT_AREA]) >= min_area:
            water[lbl == lid] = 1
    return water


def _fill_holes(mask):
    """Preenche os BURACOS FECHADOS de uma mascara binaria {0,1}.

    Flood fill do fundo a partir da borda (com 1px de padding zerado, para a
    semente nunca cair sobre o bloco): o fundo que a borda NAO alcanca esta
    cercado pelo bloco -> e buraco -> e preenchido. Assim o quarteirao fica
    cheio, sem os vaos que os carves do modelo abrem no meio dele.
    """
    m = (mask > 0).astype(np.uint8)
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8),
                  (0, 0), 1)              # preenche o fundo externo
    pad[ff == 0] = 1                      # o que sobrou em 0 era buraco fechado
    return pad[1:-1, 1:-1]


# Limiar para preenchimento de quarteiroes quase-completos: se faltam menos
# de 15% dos pixels do quarteirao para ficar todo preenchido, completa.
BLOCK_FILL_THR = 0.85       # fracao minima preenchida para completar o bloco
BLOCK_FILL_MIN_AREA = 400   # area minima do quarteirao (evita ruido)
BLOCK_FILL_MAX_AREA_FRAC = 0.15  # area maxima como fracao da imagem


def _fill_near_complete_blocks(block_full, road_prob, road_hi, veg, water,
                               typical_width, fill_thr=BLOCK_FILL_THR,
                               min_area=BLOCK_FILL_MIN_AREA):
    """Preenche quarteiroes quase-completos (regra dos 85%).

    Um QUARTEIRAO e a regiao delimitada pelas RUAS (inverso da prob de via).
    Para cada regiao entre ruas:
      1. Encontra os pixels de bloco (roof) dentro da regiao.
      2. Calcula o convex hull desses pixels.
      3. Mede a fracao do hull que ja esta preenchida.
      4. Se >= fill_thr (85%), preenche o hull inteiro — as falhas sao sombras.

    A medicao pelo CONVEX HULL (e nao pela area total da regiao) e crucial:
    buracos abertos (que conectam a rua) inflam a area total e derrubam a
    fracao, mas o hull mede exatamente 'quanto falta para fechar o poligono
    dos telhados', que e o que o usuario quer.

    Seguranca:
      - Nunca preenche sobre rua, vegetacao ou agua.
      - Exige area minima e formato 2-D.
    """
    h, w = block_full.shape
    filled = block_full.copy()

    if road_prob is None and road_hi is None:
        return filled

    # Delimita quarteiroes com road_prob (limiar moderado, dilatacao menor
    # que road_hi para nao comer bordas dos blocos).
    if road_prob is not None:
        r_thin = max(3, int(round(0.6 * typical_width)))
        road_boundary = cv2.dilate(
            (road_prob > 0.12).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * r_thin + 1, 2 * r_thin + 1)))
    else:
        road_boundary = road_hi

    # QUARTEIROES = regioes entre ruas
    block_mask = np.ones((h, w), np.uint8)
    block_mask[road_boundary > 0] = 0
    block_mask[veg > 0] = 0
    block_mask[water > 0] = 0

    n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(block_mask,
                                                            connectivity=4)

    max_area = int(BLOCK_FILL_MAX_AREA_FRAC * h * w)
    n_filled = 0
    _dbg = bool(__import__("os").environ.get("DBG_BLK"))

    for cid in range(1, n_cc):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        cw = int(stats[cid, cv2.CC_STAT_WIDTH])
        ch = int(stats[cid, cv2.CC_STAT_HEIGHT])
        if max(cw, ch) / max(min(cw, ch), 1) >= 6.0:
            continue

        x0 = int(stats[cid, cv2.CC_STAT_LEFT])
        y0 = int(stats[cid, cv2.CC_STAT_TOP])

        # Pixels de bloco/roof DENTRO desta regiao
        block_roi = (lbl[y0:y0+ch, x0:x0+cw] == cid)
        roof_roi = block_full[y0:y0+ch, x0:x0+cw]
        roof_in_block = (roof_roi > 0) & block_roi

        roof_count = int(roof_in_block.sum())
        if roof_count < min_area:
            continue

        # Convex hull dos pixels roof DENTRO do bloco
        pts = np.argwhere(roof_in_block)  # (N, 2) com (row, col)
        if len(pts) < 3:
            continue
        # cv2.convexHull quer pontos (x, y)
        pts_xy = pts[:, ::-1].astype(np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts_xy)
        if hull is None or len(hull) < 3:
            continue

        # Mascara do hull (coordenadas locais da ROI)
        hull_mask = np.zeros((ch, cw), np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        # Restringir ao bloco (nao invadir blocos vizinhos)
        hull_mask &= block_roi.astype(np.uint8)
        hull_area = int(hull_mask.sum())
        if hull_area < min_area:
            continue

        # Pixels protegidos (vegetacao, agua, ou estrada)
        veg_roi = veg[y0:y0+ch, x0:x0+cw]
        water_roi = water[y0:y0+ch, x0:x0+cw]
        protect_roi = (veg_roi > 0) | (water_roi > 0)
        if road_hi is not None:
            road_hi_roi = road_hi[y0:y0+ch, x0:x0+cw]
            protect_roi |= (road_hi_roi > 0)

        # Fracao do hull PREENCHIVEL ja preenchida
        fillable_hull = hull_mask & ~protect_roi
        fillable_area = int(fillable_hull.sum())
        filled_in_hull = int((roof_roi & fillable_hull).sum())
        frac = filled_in_hull / max(fillable_area, 1)

        if _dbg:
            print(f"  BLK cid={cid} area={area} hull={hull_area} "
                  f"fillable={fillable_area} roof={roof_count} frac={frac:.2f}")

        if frac < fill_thr:
            continue

        # Preenche o hull inteiro (coordenadas globais)
        hull_global = hull.copy()
        hull_global[:, 0, 0] += x0
        hull_global[:, 0, 1] += y0
        fill_patch = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(fill_patch, hull_global, 1)

        # Protecao: nunca preenche sobre rua/vegetacao/agua
        protect = (veg > 0) | (water > 0)
        if road_hi is not None:
            protect |= (road_hi > 0)
        fill_patch[protect] = 0

        filled[fill_patch > 0] = 1
        n_filled += 1

    if n_filled > 0:
        print(f"  Quarteiroes quase-completos preenchidos: {n_filled} "
              f"(limiar {fill_thr*100:.0f}%)")
    return filled


def _roof_mask(img_rgb, hsv, typical_width, veg, water, road_prob=None):
    """
    Telhados / quarteiroes residenciais (regioes construidas).

    Principio (ataca os 3 problemas de QA de uma vez):
      - um QUARTEIRAO/telhado e uma regiao 2-D GRANDE de superficie construida
        (telha quente, concreto branco, ou cinza de laje/telhado);
      - uma RUA e fina e alongada (1-D) -> excluida pela razao de aspecto;
      - um CARRO esta SOBRE a rua -> excluido por cair na zona de alta
        probabilidade de via do modelo (road_prob), nao no interior do bloco.

    Assim passamos a marcar o INTERIOR dos quarteiroes (antes pegava poucos
    telhados) sem marcar carros na rua (antes viravam telhado), e sem cobrir
    as ruas (o proprio sinal do modelo as recorta).
    """
    h, w = img_rgb.shape[:2]
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    warm = ((hue <= 25) | (hue >= 168)) & (sat >= 55) & (val >= 50)
    bright_white = (val >= 195) & (sat <= 45)
    gray_built = (val >= 70) & (val <= 235) & (sat <= 65)  # laje/telha/concreto
    cand = (warm | bright_white | gray_built).astype(np.uint8)
    cand[(veg > 0) | (water > 0)] = 0

    # Zona de RUA do modelo: recorta as ruas (e carros sobre elas) do candidato
    # a telhado. Limiar BAIXO (0.10) + dilatacao generosa (1.5x largura) para
    # pegar tambem as PONTAS de via, onde a probabilidade desbota abaixo de 0.20
    # — senao a ponta da rua de asfalto (cinza, cai no gray_built) virava
    # telhado vermelho.
    road_hi = None
    if road_prob is not None:
        # Ajustado para limiar moderado (0.25) e raio menor (0.7x largura tipica)
        # para evitar comer o quarteirao residencial e deixar buracos gigantes
        r = max(4, int(round(0.7 * typical_width)))
        road_hi = cv2.dilate((road_prob > 0.25).astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (2 * r + 1, 2 * r + 1)))
        cand[road_hi > 0] = 0

    cand = cv2.morphologyEx(
        cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cand = cv2.morphologyEx(
        cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # So componentes GRANDES e 2-D (quarteiroes). Carro pequeno -> fora.
    min_area = max(150, int(10.0 * typical_width * typical_width))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    roof = np.zeros((h, w), np.uint8)
    for lid in range(1, n):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bw = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh = int(stats[lid, cv2.CC_STAT_HEIGHT])
        # alongado e fino = cara de rua, nao de quarteirao
        if max(bw, bh) / max(min(bw, bh), 1) >= 4.0:
            continue
        roof[lbl == lid] = 1

    # --- Absorve SOMBRAS internas por agrupamento de cor (claro x escuro) ---
    # Depois de reconhecer os quarteiroes, nos pixels que SOBRARAM (nao bloco,
    # nao vegetacao/agua, e — importante — NAO rua, ja recortada por road_hi)
    # separa-se claro x escuro por Otsu no brilho (sem parametro fixo). O grupo
    # ESCURO e sombra de sol/predio que impede o quarteirao de fechar como
    # poligono: anexa-se ao bloco. Como as ruas foram excluidas ANTES do
    # agrupamento, o escuro restante e sombra, nunca asfalto -> nenhuma via
    # real e perdida (era o problema do fechamento morfologico anterior).
    remaining = (roof == 0) & (veg == 0) & (water == 0)
    if road_hi is not None:
        remaining &= (road_hi == 0)
    vals = val[remaining]
    if vals.size > 100:
        thr, _ = cv2.threshold(vals.reshape(-1, 1).astype(np.uint8), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = remaining & (val <= thr)
        # so a sombra ADJACENTE a um quarteirao (nao manchas escuras isoladas)
        r2 = max(4, int(round(1.5 * typical_width)))
        near = cv2.dilate(roof, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * r2 + 1, 2 * r2 + 1))) > 0
        roof[dark & near] = 1

    roof = cv2.morphologyEx(
        roof, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # PREENCHE OS QUARTEIROES (logica de "quarteirao solido"): sombras, vaos e os
    # carves do modelo deixavam BURACOS/baias no bloco. Um CLOSE grande junta o
    # bloco fragmentado (atravessa as baias e sombras) e o preenche-buracos
    # fecha o miolo -> quarteirao cheio. Esse bloco solido e a base tanto da viz
    # quanto do filtro; cada um REABRE as ruas no seu limiar (ver abaixo).
    kbig = 2 * max(4, int(round(1.5 * typical_width))) + 1
    block_full = _fill_holes(cv2.morphologyEx(
        roof, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                          (kbig, kbig))))

    # QUARTEIROES QUASE-COMPLETOS: se faltam menos de 15% do quarteirao
    # (delimitado pelas ruas) para ficar todo preenchido, completa com
    # vermelho. As falhas sao sombras. Roda APOS o block_full para medir a
    # fracao sobre os blocos ja consolidados (CLOSE grande + fill_holes).
    block_full = _fill_near_complete_blocks(
        block_full, road_prob, road_hi, veg, water, typical_width)

    # VIZ / supressao: reabre as ruas CONFIAVEIS (road_hi, limiar baixo 0.10),
    # para o vermelho do quarteirao nunca cobrir uma via — mas sem buracos.
    roof = block_full.copy()
    if road_hi is not None:
        roof[road_hi > 0] = 0

    # FILTRO de grafo: reabre SO o GRID FORTE (prob>0.30). O toco interno fraco
    # continua COBERTO (a ponta acusa "dentro do bloco" e e removido), enquanto a
    # rua real (forte) fica fora do bloco e e preservada. Sem road_prob, usa roof.
    roof_solid = roof.copy()
    if road_prob is not None:
        strong = cv2.dilate(
            (road_prob > 0.30).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kbig, kbig)))
        roof_solid = block_full.copy()
        roof_solid[strong > 0] = 0
    return roof, roof_solid


def _soil_mask(img_rgb, hsv, typical_width, veg, water, roof):
    """Solo exposto: paleta de terra em mancha 2-D (nao-linear)."""
    h, w = img_rgb.shape[:2]
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    warm_soil = (hue >= 5) & (hue <= 30) & (sat >= 35) & (val >= 80)
    cand = warm_soil.astype(np.uint8)
    cand[(veg > 0) | (water > 0) | (roof > 0)] = 0
    cand = cv2.morphologyEx(
        cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    min_area = max(1500, int(20.0 * typical_width * typical_width))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    soil = np.zeros((h, w), np.uint8)
    for lid in range(1, n):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0 = int(stats[lid, cv2.CC_STAT_LEFT]);   y0 = int(stats[lid, cv2.CC_STAT_TOP])
        cw = int(stats[lid, cv2.CC_STAT_WIDTH]);  ch = int(stats[lid, cv2.CC_STAT_HEIGHT])
        comp = (lbl[y0:y0 + ch, x0:x0 + cw] == lid)
        if skeletonize is None:
            continue
        skel_len = int(skeletonize(comp).sum())
        # Estrada (1-D): comprimento^2/area = L/w >> 1. Clareira (2-D): ~1.
        ratio = (skel_len * skel_len) / max(area, 1)
        if ratio < 6.0:
            soil[y0:y0 + ch, x0:x0 + cw][comp] = 1
    return soil


BRIGHT_VMIN = 205  # limiar de brilho padrao (quase-branco)


def _bright_pavement_mask(hsv, vmin=BRIGHT_VMIN):
    """
    Superficie CLARA e NEUTRA (val alto, sat baixa): calcada/concreto claro,
    bordas claras de telhado, corredores claros. Em ZOOM PROXIMO/URBANO uma
    rua de veiculo de verdade e asfalto ESCURO; uma linha clara raramente e
    rua. (Em satelite alto/rural, estrada de terra clara e valida -> este
    portao fica DESLIGADO nesses casos.)
    Nao inclui terra quente (sat mais alta) nem asfalto escuro (val baixo).

    vmin: limiar de brilho. MENOR = mais agressivo (rejeita mais claros);
    MAIOR = mais tolerante. Padrao 205 (quase-branco) nao come o cinza-medio
    das ruas reais; abaixo de ~150 comeca a comer rua de verdade.
    """
    val = hsv[:, :, 2].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)
    return ((val > int(vmin)) & (sat < 40)).astype(np.uint8)


def is_rural(veg_frac, road_density):
    """Rural = muito verde E malha viaria esparsa (favela em vale verde tem
    muito verde mas malha densa -> NAO e rural)."""
    return veg_frac > ROOF_SKIP_VEG and road_density < ROAD_DENSITY_RURAL


def build_context_masks(img_rgb, typical_width=8.0, road_prob=None,
                        reject_bright=False, bright_vmin=BRIGHT_VMIN,
                        road_density=1.0):
    """
    Retorna dict de mascaras uint8 {0,1}: vegetation, water, roof, soil,
    bright_reject, e 'suppression' (float32 0..1) multiplicador do mapa de prob.

    road_prob (opcional): mapa de probabilidade de via do modelo. Quando
    fornecido, as RUAS sao recortadas dos quarteiroes/telhados — impede
    marcar carros na rua como telhado e cobrir vias com a mascara de bloco.

    reject_bright: liga o portao de brilho (suprime vias CLARAS/neutras). Usado
    em zoom proximo/urbano, onde a rua real e escura. Diferente das outras
    mascaras, ele e aplicado de forma DURA (sobre a probabilidade alta tambem),
    pois a queixa e justamente de deteccoes claras CONFIANTES.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    std_map = _local_std(gray, 7)

    veg = _vegetation_mask(img_rgb, hsv)
    water = _water_mask(img_rgb, hsv, std_map)

    # RURAL (muito verde E malha esparsa): nao usa telhado/bloco nem solo — eles
    # so pintariam campos de vermelho/amarelo e poderiam suprimir estrada de
    # terra no campo. Favela em vale verde (malha densa) NAO entra aqui.
    rural = is_rural(float(veg.mean()), road_density)
    if rural:
        roof = np.zeros(img_rgb.shape[:2], np.uint8)
        roof_solid = roof
        soil = np.zeros(img_rgb.shape[:2], np.uint8)
    else:
        roof, roof_solid = _roof_mask(
            img_rgb, hsv, typical_width, veg, water, road_prob)
        soil = _soil_mask(img_rgb, hsv, typical_width, veg, water, roof)
    bright = (_bright_pavement_mask(hsv, bright_vmin) if reject_bright
              else np.zeros(img_rgb.shape[:2], np.uint8))

    # Telhado e solo sao supressao SUAVE (mantem um residual): assim um trecho
    # de RUA fraco que caia sobre telhado/solo (asfalto cinza = mesma cor de
    # laje) sobrevive e conecta a malha, em vez de sumir. Vegetacao e agua sao
    # supressao dura (definitivamente nao sao via). Pixels de alta confianca do
    # modelo nunca sao suprimidos (ver apply_suppression).
    suppression = np.ones(img_rgb.shape[:2], np.float32)
    suppression[roof > 0] = ROOF_KEEP
    suppression[soil > 0] = SOIL_KEEP
    suppression[veg > 0] = 0.0
    suppression[water > 0] = 0.0

    return {
        "vegetation": veg,
        "water": water,
        "roof": roof,
        "roof_solid": roof_solid,  # so p/ o filtro de bloco (tocos internos)
        "soil": soil,
        "bright_reject": bright,
        "suppression": suppression,
    }


def apply_suppression(road_prob, masks, protect_thr=PROTECT_THR):
    """
    Multiplica o mapa de probabilidade pela supressao de contexto,
    preservando pixels em que o modelo tem alta confianca — EXCETO o portao de
    brilho (bright_reject), que e aplicado de forma dura (zera mesmo onde a
    probabilidade e alta), pois a queixa e de vias claras confiantes.
    """
    suppressed = road_prob * masks["suppression"]
    out = np.where(road_prob >= protect_thr, road_prob, suppressed)
    bright = masks.get("bright_reject")
    if bright is not None and bright.any():
        out = np.where(bright > 0, 0.0, out)
    return out


def masks_debug_image(img_rgb, masks):
    """Visualizacao para o relatorio: verde=vegetacao, azul=agua,
    vermelho=telhado, amarelo=solo exposto."""
    viz = img_rgb.copy()
    overlay = img_rgb.copy()
    overlay[masks["vegetation"] > 0] = [40, 180, 40]
    overlay[masks["soil"] > 0] = [230, 210, 40]
    overlay[masks["roof"] > 0] = [220, 40, 40]
    overlay[masks["water"] > 0] = [40, 80, 230]
    if masks.get("bright_reject") is not None:
        overlay[masks["bright_reject"] > 0] = [230, 40, 230]  # magenta
    return cv2.addWeighted(viz, 0.45, overlay, 0.55, 0)
